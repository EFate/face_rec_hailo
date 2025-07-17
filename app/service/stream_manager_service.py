# app/service/stream_manager_service.py
import asyncio
import multiprocessing as mp
import queue
import threading
import os
import uuid
from typing import List, Dict, Any
from datetime import datetime, timedelta
from fastapi import HTTPException, status

from app.cfg.config import AppSettings
from app.cfg.logging import app_logger
from app.core.pipeline import FaceStreamPipeline
from app.schema.face_schema import ActiveStreamInfo, StreamStartRequest

# 特殊信号定义
SUCCESS_SIGNAL = "__SUCCESS__"

def video_stream_process_worker(
        stream_id: str,
        video_source: str,
        settings_dict: Dict[str, Any],
        main_app_pid: int,
        result_queue: mp.Queue,
        stop_event: mp.Event
):
    """
    视频处理工作函数 (子进程入口点)。
    """
    if video_source.startswith("rtsp://"):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        app_logger.info(f"【子进程 {stream_id}】检测到RTSP源，已设置强制TCP传输。")

    pipeline = None
    try:
        settings = AppSettings.model_validate(settings_dict)
        pipeline = FaceStreamPipeline(
            settings=settings,
            stream_id=stream_id,
            video_source=video_source,
            main_app_pid=main_app_pid,
            output_queue=result_queue
        )
        result_queue.put_nowait(SUCCESS_SIGNAL)
        app_logger.info(f"【子进程 {stream_id}】初始化成功，已发送启动信号。")
        
        def on_stop():
            stop_event.wait()
            if pipeline:
                pipeline.stop()

        stop_monitor = threading.Thread(target=on_stop, daemon=True)
        stop_monitor.start()
        pipeline.start()

    except Exception as e:
        app_logger.error(f"【子进程 {stream_id}】发生致命错误: {e}", exc_info=True)
        try:
            result_queue.put_nowait(e)
        except (queue.Full, ValueError):
            pass
    finally:
        try:
            result_queue.put_nowait(None)
        except (queue.Full, ValueError, OSError):
            pass
        app_logger.info(f"✅【子进程 {stream_id}】处理工作已结束。")


class StreamManagerService:
    """
    负责管理视频流的生命周期，包括启动、停止、状态监控和后台清理。
    """
    def __init__(self, settings: AppSettings):
        app_logger.info("正在初始化 StreamManagerService...")
        self.settings = settings
        self.active_streams: Dict[str, Dict[str, Any]] = {}
        self.stream_lock = asyncio.Lock()
        self.mp_context = mp.get_context("spawn")

    async def initialize(self):
        """异步初始化（如果需要）。"""
        app_logger.info("✅ StreamManagerService 初始化完毕。")
        pass

    async def start_stream(self, req: StreamStartRequest) -> ActiveStreamInfo:
        stream_id = str(uuid.uuid4())
        lifetime = req.lifetime_minutes if req.lifetime_minutes is not None else self.settings.app.stream_default_lifetime_minutes
        
        result_queue = self.mp_context.Queue(maxsize=30)
        stop_event = self.mp_context.Event()

        main_app_pid = os.getpid()
        app_logger.info(f"主应用PID为 {main_app_pid}，将传递给子进程 {stream_id}。")

        process = self.mp_context.Process(
            target=video_stream_process_worker,
            args=(stream_id, req.source, self.settings.model_dump(), main_app_pid, result_queue, stop_event),
            daemon=True
        )
        process.start()

        try:
            init_signal = await asyncio.to_thread(result_queue.get, timeout=20.0)
            if isinstance(init_signal, Exception):
                raise init_signal
            elif init_signal != SUCCESS_SIGNAL:
                raise RuntimeError(f"收到来自子进程的未知启动信号: {init_signal}")
            app_logger.info(f"✅ 主进程收到来自 {stream_id} 的成功启动信号。")
        except queue.Empty:
            err_msg = f"启动视频流任务超时（20秒），请检查视频源 '{req.source}' 是否可用或网络连接是否正常。"
            app_logger.error(f"启动视频流 {stream_id} 失败: {err_msg}")
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
            raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=err_msg)
        except Exception as e:
            err_msg = f"无法启动视频流，子进程初始化失败: {e}"
            app_logger.error(f"启动视频流 {stream_id} 失败: {err_msg}", exc_info=False)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=err_msg)

        async with self.stream_lock:
            started_at = datetime.now()
            expires_at = None if lifetime == -1 else started_at + timedelta(minutes=lifetime)
            stream_info = ActiveStreamInfo(stream_id=stream_id, source=req.source, started_at=started_at,
                                           expires_at=expires_at, lifetime_minutes=lifetime)
            self.active_streams[stream_id] = {
                "info": stream_info, "queue": result_queue, "stop_event": stop_event, "process": process
            }
            app_logger.info(f"🚀 视频流进程已启动并验证成功: ID={stream_id}, Source={req.source}")
            return stream_info

    async def stop_stream(self, stream_id: str) -> bool:
        async with self.stream_lock:
            stream_context = self.active_streams.pop(stream_id, None)
            if not stream_context:
                return False

        process = stream_context["process"]
        if process.is_alive():
            app_logger.info(f"正在向视频流 {stream_id} 发送停止信号...")
            stream_context["stop_event"].set()
            await asyncio.to_thread(process.join, timeout=5.0)
            if process.is_alive():
                app_logger.warning(f"视频流 {stream_id} 未能优雅退出，正在强制终止。")
                process.terminate()
                await asyncio.to_thread(process.join, timeout=2.0)
        
        queue_to_close = stream_context["queue"]
        queue_to_close.close()
        try:
            await asyncio.to_thread(queue_to_close.join_thread)
        except (AttributeError, ValueError, OSError):
            pass

        app_logger.info(f"✅ 视频流已成功停止并清理: ID={stream_id}")
        return True

    async def get_stream_feed(self, stream_id: str):
        try:
            async with self.stream_lock:
                if stream_id not in self.active_streams:
                    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Stream not found.")
                stream_data = self.active_streams[stream_id]
                frame_queue = stream_data["queue"]
                process = stream_data["process"]

            while True:
                if not process.is_alive() and frame_queue.empty():
                    app_logger.warning(f"流 {stream_id} 的工作进程已终止且队列为空，停止推送。")
                    break
                
                try:
                    frame_bytes = await asyncio.to_thread(frame_queue.get, timeout=0.01)
                    if frame_bytes is None:
                        break 
                    if frame_bytes == SUCCESS_SIGNAL:
                        continue 

                    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                except queue.Empty:
                    await asyncio.sleep(0.01)
                    continue
                except (ValueError, OSError):
                    app_logger.info(f"流 {stream_id} 的队列已被关闭，正常停止推送。")
                    break

        except asyncio.CancelledError:
            app_logger.info(f"客户端从流 {stream_id} 断开连接。")
        except Exception as e:
            app_logger.error(f"推送流 {stream_id} 时发生未知错误: {e}", exc_info=True)
        finally:
            app_logger.info(f"已停止向客户端推送流 {stream_id} 的数据。")

    async def get_all_active_streams_info(self) -> List[ActiveStreamInfo]:
        async with self.stream_lock:
            if not self.active_streams: return []
            
            active_infos = []
            dead_stream_ids = []

            for stream_id, stream in list(self.active_streams.items()):
                if stream["process"].is_alive():
                    active_infos.append(stream["info"])
                else:
                    app_logger.warning(f"检测到视频流进程 {stream_id} 已意外终止，将进行清理。")
                    dead_stream_ids.append(stream_id)
            
            if dead_stream_ids:
                app_logger.info(f"正在清理已终止的流: {dead_stream_ids}")
                for sid in dead_stream_ids:
                    dead_stream_data = self.active_streams.pop(sid, None)
                    if dead_stream_data:
                        q = dead_stream_data["queue"]
                        q.close()
                        try:
                           q.join_thread()
                        except (AttributeError, ValueError, OSError):
                           pass
            return active_infos

    async def cleanup_expired_streams(self):
        while True:
            await asyncio.sleep(self.settings.app.stream_cleanup_interval_seconds)
            now = datetime.now()
            
            async with self.stream_lock:
                stream_items = list(self.active_streams.items())

            expired_ids = [
                sid for sid, s_ctx in stream_items
                if s_ctx["info"].expires_at and now >= s_ctx["info"].expires_at
            ]

            if expired_ids:
                app_logger.info(f"后台任务检测到过期视频流: {expired_ids}")
                await asyncio.gather(*[self.stop_stream(sid) for sid in expired_ids])

    async def stop_all_streams(self):
        async with self.stream_lock:
            if not self.active_streams: return
            all_ids = list(self.active_streams.keys())
        
        if all_ids:
            app_logger.info(f"正在停止所有活动流: {all_ids}")
            await asyncio.gather(*[self.stop_stream(sid) for sid in all_ids])