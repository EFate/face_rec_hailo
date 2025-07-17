# app/core/pipeline.py
import queue
import threading
import time
from typing import List, Dict, Any, Optional, Set
import os
import gc

import cv2
import numpy as np
import degirum as dg

from app.cfg.config import AppSettings
from app.cfg.logging import app_logger
from app.core.model_manager import create_degirum_model, DeGirumModel
from app.service.face_dao import LanceDBFaceDataDAO, FaceDataDAO
from app.core.image_utils import align_and_crop
from .process_utils import get_degirum_worker_pids, cleanup_degirum_workers_by_pids


def _draw_results_on_frame(frame: np.ndarray, results: List[Dict[str, Any]]):
    """在帧上绘制识别结果"""
    for res in results:
        box = res.get('box')
        if not box: continue
        label = f"{res.get('name', 'Unknown')}"
        similarity = res.get('similarity')
        if similarity is not None and label != "Unknown":
            label += f" ({similarity:.2f})"
        color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (box[0], box[1] - lh - 10), (box[0] + lw, box[1]), color, cv2.FILLED)
        cv2.putText(frame, label, (box[0] + 5, box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


class FaceStreamPipeline:
    def __init__(self, settings: AppSettings, stream_id: str, video_source: str, main_app_pid: int, output_queue: queue.Queue):
        self.settings = settings
        self.stream_id = stream_id
        self.video_source = video_source
        self.output_queue = output_queue
        self.main_app_pid = main_app_pid
        self.my_worker_pids: Set[int] = set()

        app_logger.info(f"【流水线 {self.stream_id}】正在初始化，关联的主应用PID为 {self.main_app_pid}...")
        try:
            source_for_cv = int(self.video_source) if self.video_source.isdigit() else self.video_source
            self.cap = cv2.VideoCapture(source_for_cv)
            if not self.cap.isOpened():
                raise RuntimeError(f"无法打开视频源: {self.video_source}")
        except Exception as e:
            app_logger.error(f"【流水线 {self.stream_id}】初始化视频源失败: {e}")
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()
            raise

        app_logger.info(f"【流水线 {self.stream_id}】准备加载模型，记录初始worker状态...")
        workers_before = get_degirum_worker_pids(self.main_app_pid)
        app_logger.info(f"【流水线 {self.stream_id}】初始worker PIDs: {workers_before}")

        self.det_model: DeGirumModel = create_degirum_model(self.settings.degirum.detection_model_name, self.settings.degirum.zoo_url)
        self.rec_model: DeGirumModel = create_degirum_model(self.settings.degirum.recognition_model_name, self.settings.degirum.zoo_url)
        
        app_logger.info(f"【流水线 {self.stream_id}】模型加载完毕，记录结束worker状态...")
        workers_after = get_degirum_worker_pids(self.main_app_pid)
        app_logger.info(f"【流水线 {self.stream_id}】结束worker PIDs: {workers_after}")

        self.my_worker_pids = workers_after - workers_before
        if not self.my_worker_pids:
            app_logger.warning(f"【流水线 {self.stream_id}】未检测到任何新的专属工作进程。")
        else:
            app_logger.warning(f"【流水线 {self.stream_id}】已识别出专属的DeGirum工作进程 PIDs: {self.my_worker_pids}")
        
        self.face_dao: FaceDataDAO = LanceDBFaceDataDAO(db_uri=self.settings.degirum.lancedb_uri, table_name=self.settings.degirum.lancedb_table_name)
        self.preprocess_queue, self.inference_queue, self.postprocess_queue = queue.Queue(maxsize=10), queue.Queue(maxsize=10), queue.Queue(maxsize=10)
        self.stop_event = threading.Event()
        self.threads: List[threading.Thread] = []

    def stop(self):
        if self.stop_event.is_set():
            return
        app_logger.info(f"【流水线 {self.stream_id}】正在停止...")
        self.stop_event.set()

        for t in self.threads:
            if t.is_alive():
                t.join(timeout=2.0)
        if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
            self.cap.release()
        for q in [self.preprocess_queue, self.inference_queue, self.postprocess_queue]:
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
        if hasattr(self, 'face_dao'):
            self.face_dao.dispose()

        app_logger.info(f"【流水线 {self.stream_id}】开始执行模型资源强制释放流程...")
        if hasattr(self, 'det_model') and self.det_model:
            del self.det_model
        if hasattr(self, 'rec_model') and self.rec_model:
            del self.rec_model
        
        app_logger.warning(f"【流水线 {self.stream_id}】执行靶向清理程序...")
        cleanup_degirum_workers_by_pids(self.my_worker_pids, app_logger)
        
        gc.collect()
        app_logger.info(f"✅【流水线 {self.stream_id}】所有资源已清理，已安全停止。")

    def start(self):
        app_logger.info(f"【流水线 {self.stream_id}】正在启动...")
        try:
            self._start_threads()
            while not self.stop_event.is_set():
                if not all(t.is_alive() for t in self.threads):
                    app_logger.error(f"❌【流水线 {self.stream_id}】检测到有工作线程意外终止，正在停止整个流水线。")
                    break
                time.sleep(1)
        except Exception as e:
            app_logger.error(f"❌【流水线 {self.stream_id}】启动或运行时失败: {e}", exc_info=True)
        finally:
            self.stop()
            
    def _start_threads(self):
        self.threads = [
            threading.Thread(target=self._reader_thread, name=f"Reader-{self.stream_id}", daemon=True),
            threading.Thread(target=self._preprocessor_thread, name=f"Preprocessor-{self.stream_id}", daemon=True),
            threading.Thread(target=self._inference_thread, name=f"Inference-{self.stream_id}", daemon=True),
            threading.Thread(target=self._postprocessor_thread, name=f"Postprocessor-{self.stream_id}", daemon=True)
        ]
        for t in self.threads:
            t.start()

    def _reader_thread(self):
        app_logger.info(f"【T1:读帧 {self.stream_id}】启动。")
        while not self.stop_event.is_set():
            if not (hasattr(self, 'cap') and self.cap and self.cap.isOpened()):
                app_logger.warning(f"【T1:读帧 {self.stream_id}】摄像头未打开或已释放，线程终止。")
                break
            ret, frame = self.cap.read()
            if not ret:
                app_logger.warning(f"【T1:读帧 {self.stream_id}】无法读取帧，流可能已结束。")
                time.sleep(1)
                if self.stop_event.is_set():
                    break
                continue
            if self.preprocess_queue.full():
                try:
                    self.preprocess_queue.get_nowait()
                except queue.Empty:
                    pass
            self.preprocess_queue.put(frame)
            time.sleep(0.01)
        self.preprocess_queue.put(None)
        app_logger.info(f"【T1:读帧 {self.stream_id}】已停止。")

    def _preprocessor_thread(self):
        app_logger.info(f"【T2:预处理 {self.stream_id}】启动。")
        while not self.stop_event.is_set():
            try:
                frame = self.preprocess_queue.get(timeout=1.0)
                if frame is None:
                    self.inference_queue.put(None)
                    break
                self.inference_queue.put(frame)
            except queue.Empty:
                continue
        app_logger.info(f"【T2:预处理 {self.stream_id}】已停止。")

    def _inference_thread(self):
        app_logger.info(f"【T3:推理-检测 {self.stream_id}】启动。")
        while not self.stop_event.is_set():
            try:
                frame = self.inference_queue.get(timeout=1.0)
                if frame is None:
                    self.postprocess_queue.put(None)
                    break
                detection_results = []
                if hasattr(self, 'det_model') and self.det_model:
                    try:
                        detection_result_obj = self.det_model.predict(frame)
                        detection_results = detection_result_obj.results
                    except Exception as model_exc:
                        app_logger.error(f"【T3:推理-检测 {self.stream_id}】模型 'predict' 调用失败: {model_exc}", exc_info=False)
                self.postprocess_queue.put((frame, detection_results))
            except queue.Empty:
                continue
            except Exception as e:
                app_logger.error(f"【T3:推理-检测 {self.stream_id}】发生严重错误: {e}", exc_info=True)
                break 
        self.postprocess_queue.put(None)
        app_logger.info(f"【T3:推理-检测 {self.stream_id}】已停止。")

    def _postprocessor_thread(self):
        app_logger.info(f"【T4:后处理-识别 {self.stream_id}】启动。")
        threshold = self.settings.degirum.recognition_similarity_threshold
        while not self.stop_event.is_set():
            try:
                data = self.postprocess_queue.get(timeout=1.0)
                if data is None:
                    break
                original_frame, detected_faces_data = data
                final_results = []
                if detected_faces_data:
                    aligned_faces, valid_faces_meta = [], []
                    for face_data in detected_faces_data:
                        landmarks = [lm["landmark"] for lm in face_data.get("landmarks", [])]
                        if len(landmarks) == 5:
                            aligned_face, _ = align_and_crop(original_frame, landmarks)
                            if aligned_face.size > 0:
                                aligned_faces.append(aligned_face)
                                valid_faces_meta.append(face_data)
                    if aligned_faces and hasattr(self, 'rec_model') and self.rec_model:
                        try:
                            batch_rec_results = self.rec_model.predict_batch(aligned_faces)
                            for i, rec_result in enumerate(batch_rec_results):
                                embedding = np.array(rec_result.results[0]['data'][0])
                                face_meta = valid_faces_meta[i]
                                search_res = self.face_dao.search(embedding, threshold)
                                result_item = {"box": list(map(int, face_meta['bbox'])), "name": "Unknown", "similarity": None}
                                if search_res:
                                    name, sn, similarity = search_res
                                    result_item.update({"name": name, "sn": sn, "similarity": similarity})
                                final_results.append(result_item)
                        except Exception as model_exc:
                            app_logger.error(f"【T4:后处理-识别 {self.stream_id}】模型 'predict_batch' 调用失败: {model_exc}", exc_info=False)
                            for face_meta in valid_faces_meta:
                                final_results.append({"box": list(map(int, face_meta['bbox'])), "name": "Unknown", "similarity": None})
                
                _draw_results_on_frame(original_frame, final_results)
                (flag, encodedImage) = cv2.imencode(".jpg", original_frame)
                if flag:
                    try:
                        self.output_queue.put_nowait(encodedImage.tobytes())
                    except (queue.Full, ValueError):
                        pass 
            except queue.Empty:
                continue
            except Exception as e:
                app_logger.error(f"【T4:后处理-识别 {self.stream_id}】发生严重错误: {e}", exc_info=True)
                break
        try:
            self.output_queue.put_nowait(None) 
        except (queue.Full, ValueError, OSError):
            pass
        app_logger.info(f"【T4:后处理-识别 {self.stream_id}】已停止。")