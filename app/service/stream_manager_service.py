# app/service/stream_manager_service.py
import asyncio
import queue
import threading
import uuid
from typing import List, Dict, Any
from datetime import datetime, timedelta

from fastapi import HTTPException, status

from app.cfg.config import AppSettings
from app.cfg.logging import app_logger
from app.core.pipeline import FaceStreamPipeline
from app.schema.face_schema import ActiveStreamInfo, StreamStartRequest
# ❗【修改】导入 ModelPool
from app.core.model_manager import ModelPool

class StreamManagerService:
    """
    【核心修改】负责管理视频流的生命周期，使用线程模型，并将模型池注入每个管道。
    """
    def __init__(self, settings: AppSettings, model_pool: ModelPool):
        app_logger.info("正在初始化 StreamManagerService (使用线程+模型池)...")
        self.settings = settings
        # ❗【修改】持有对模型池的引用
        self.model_pool = model_pool
        self.active_streams: Dict[str, Dict[str, Any]] = {}
        self.stream_lock = asyncio.Lock()

    async def start_stream(self, req: StreamStartRequest) -> ActiveStreamInfo:
        stream_id = str(uuid.uuid4())
        lifetime = req.lifetime_minutes if req.lifetime_minutes is not None else self.settings.app.stream_default_lifetime_minutes
        
        app_logger.info(f"准备为流 {stream_id} 启动一个新线程 (它将从池中获取模型)...")
        
        try:
            # 1. 创建线程安全的队列
            frame_queue = queue.Queue(maxsize=30)
            
            # 2. ❗【修改】实例化流水线，注入模型池
            pipeline = FaceStreamPipeline(
                settings=self.settings,
                stream_id=stream_id,
                video_source=req.source,
                model_pool=self.model_pool, # 注入模型池
                output_queue=frame_queue
            )

            # 3. 创建线程，目标是流水线的 start 方法
            process_thread = threading.Thread(target=pipeline.start, daemon=True)
            
        except Exception as e:
            err_msg = f"无法准备视频流处理管道: {e}"
            app_logger.error(f"启动视频流 {stream_id} 失败: {err_msg}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=err_msg)

        # 4. 启动线程
        process_thread.start()

        # 短暂等待以确认线程是否立即失败 (例如，因无法获取模型)
        await asyncio.sleep(0.2)
        if not process_thread.is_alive():
             raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "服务正忙，无法启动新的视频流，请稍后再试。")

        async with self.stream_lock:
            started_at = datetime.now()
            expires_at = None if lifetime == -1 else started_at + timedelta(minutes=lifetime)
            stream_info = ActiveStreamInfo(stream_id=stream_id, source=req.source, started_at=started_at, expires_at=expires_at, lifetime_minutes=lifetime)
            self.active_streams[stream_id] = {
                "info": stream_info, "queue": frame_queue, "pipeline": pipeline, "thread": process_thread,
            }
            app_logger.info(f"🚀 视频流处理线程已启动: ID={stream_id}, Source={req.source}")
            return stream_info

    async def stop_stream(self, stream_id: str) -> bool:
        async with self.stream_lock:
            stream_context = self.active_streams.pop(stream_id, None)
            if not stream_context: return False

        pipeline: FaceStreamPipeline = stream_context["pipeline"]
        thread: threading.Thread = stream_context["thread"]
        
        if thread.is_alive():
            pipeline.stop()
            await asyncio.to_thread(thread.join, timeout=5.0)
        
        return True

    # get_stream_feed, get_all_active_streams_info, cleanup_expired_streams, stop_all_streams 
    # 这些方法的逻辑与纯线程方案一致，无需重大修改
    async def get_stream_feed(self, stream_id: str):
        # ... 此方法代码保持不变 ...
        try:
            async with self.stream_lock:
                if stream_id not in self.active_streams:
                    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Stream not found.")
                stream_data = self.active_streams[stream_id]
                frame_queue = stream_data["queue"]
                thread = stream_data["thread"]
            while True:
                if not thread.is_alive() and frame_queue.empty(): break
                try:
                    frame_bytes = await asyncio.to_thread(frame_queue.get, timeout=0.02)
                    if frame_bytes is None: break
                    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                except queue.Empty: await asyncio.sleep(0.01)
        except asyncio.CancelledError: pass
    async def get_all_active_streams_info(self) -> List[ActiveStreamInfo]:
        # ... 此方法代码保持不变 ...
        async with self.stream_lock:
            active_infos = []
            dead_stream_ids = [sid for sid, s_ctx in self.active_streams.items() if not s_ctx["thread"].is_alive()]
            for stream_id, stream in self.active_streams.items():
                if stream["thread"].is_alive(): active_infos.append(stream["info"])
            for sid in dead_stream_ids: self.active_streams.pop(sid, None)
            return active_infos
    async def cleanup_expired_streams(self):
        # ... 此方法代码保持不变 ...
        while True:
            await asyncio.sleep(self.settings.app.stream_cleanup_interval_seconds)
            now = datetime.now()
            async with self.stream_lock:
                expired_ids = [sid for sid, s_ctx in self.active_streams.items() if s_ctx["info"].expires_at and now >= s_ctx["info"].expires_at]
            if expired_ids: await asyncio.gather(*[self.stop_stream(sid) for sid in expired_ids])
    async def stop_all_streams(self):
        # ... 此方法代码保持不变 ...
        async with self.stream_lock: all_ids = list(self.active_streams.keys())
        if all_ids: await asyncio.gather(*[self.stop_stream(sid) for sid in all_ids])