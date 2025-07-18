# app/core/pipeline.py
import queue
import threading
import time
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np

from app.cfg.config import AppSettings
from app.cfg.logging import app_logger
from app.core.model_manager import ModelPool, DeGirumModel
from app.core.image_utils import align_and_crop
from app.service.face_dao import LanceDBFaceDataDAO, FaceDataDAO

def _draw_results_on_frame(frame: np.ndarray, results: List[Dict[str, Any]]):
    """在帧上绘制识别结果 (保持不变)"""
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
    def __init__(self, settings: AppSettings, stream_id: str, video_source: str, model_pool: ModelPool, output_queue: queue.Queue):
        self.settings = settings
        self.stream_id = stream_id
        self.video_source = video_source
        self.output_queue = output_queue
        self.model_pool = model_pool
        self.models: Optional[Tuple[DeGirumModel, DeGirumModel]] = None
        self.det_model: Optional[DeGirumModel] = None
        self.rec_model: Optional[DeGirumModel] = None
        self.stop_event = threading.Event()
        self.threads: List[threading.Thread] = []
        self.face_dao: FaceDataDAO = LanceDBFaceDataDAO(db_uri=self.settings.degirum.lancedb_uri, table_name=self.settings.degirum.lancedb_table_name)
        self.preprocess_queue = queue.Queue(maxsize=30)
        self.inference_queue = queue.Queue(maxsize=30)
        self.postprocess_queue = queue.Queue(maxsize=30)

    def start(self):
        app_logger.info(f"【流水线 {self.stream_id}】正在启动，并尝试获取模型...")
        try:
            self.models = self.model_pool.acquire(timeout=5.0)
            if self.models is None:
                app_logger.error(f"❌【流水线 {self.stream_id}】启动失败：无法从模型池中获取可用模型。")
                return

            self.det_model, self.rec_model = self.models
            app_logger.info(f"【流水线 {self.stream_id}】成功获取模型，准备打开视频源...")

            source_for_cv = int(self.video_source) if self.video_source.isdigit() else self.video_source
            self.cap = cv2.VideoCapture(source_for_cv)
            if not self.cap.isOpened():
                raise RuntimeError(f"无法打开视频源: {self.video_source}")

            self._start_threads()
            
            while not self.stop_event.is_set():
                if not all(t.is_alive() for t in self.threads):
                    app_logger.error(f"❌【流水线 {self.stream_id}】检测到有工作线程意外终止。")
                    break
                time.sleep(1) 

        except Exception as e:
            app_logger.error(f"❌【流水线 {self.stream_id}】启动或运行时失败: {e}", exc_info=True)
        finally:
            self.stop()

    def stop(self):
        """【优化】确保所有资源都被正确、有序地释放。"""
        if self.stop_event.is_set(): return
        app_logger.warning(f"【流水线 {self.stream_id}】正在停止...")
        self.stop_event.set()

        # 等待所有线程结束
        for t in self.threads:
            # 超时设置得更短
            t.join(timeout=5.0)
            if t.is_alive():
                app_logger.error(f"【流水线 {self.stream_id}】线程 {t.name} 未能快速停止，可能被I/O阻塞。")

        # 释放视频捕捉对象
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
            app_logger.info(f"【流水线 {self.stream_id}】视频捕捉已释放。")

        # 清空所有中间队列
        for q in [self.preprocess_queue, self.inference_queue, self.postprocess_queue]:
            while not q.empty():
                try: q.get_nowait()
                except queue.Empty: break

        # 归还模型到池中
        if self.models:
            self.model_pool.release(self.models)
            app_logger.info(f"【流水线 {self.stream_id}】已将模型归还到池中。")
            self.models = None
            self.det_model = None
            self.rec_model = None

        app_logger.info(f"✅【流水线 {self.stream_id}】所有资源已清理。")

    def _start_threads(self):
        thread_targets = {
            "Reader": self._reader_thread,
            "Preprocessor": self._preprocessor_thread,
            "Inference": self._inference_thread,
            "Postprocessor": self._postprocessor_thread
        }
        for name, target in thread_targets.items():
            thread = threading.Thread(target=target, name=f"{self.stream_id}-{name}", daemon=True)
            self.threads.append(thread)
            thread.start()

    def _reader_thread(self):
        """
        ❗【核心优化】重写读帧线程逻辑，使用非阻塞put，确保能快速响应停止信号。
        """
        app_logger.info(f"【T1:读帧 {self.stream_id}】启动。")
        while not self.stop_event.is_set():
            if not (hasattr(self, 'cap') and self.cap.isOpened()):
                app_logger.warning(f"【T1:读帧 {self.stream_id}】视频源已关闭或不可用。")
                break

            ret, frame = self.cap.read()
            if not ret:
                is_file = not self.video_source.isdigit()
                if is_file:
                    app_logger.info(f"【T1:读帧 {self.stream_id}】视频文件已读完 (EOF)。")
                    break
                time.sleep(0.01)
                continue

            try:
                # 优化：使用非阻塞的put_nowait，避免长时间阻塞
                self.preprocess_queue.put_nowait(frame)
            except queue.Full:
                # 当下游处理慢导致队列满时，丢弃帧并立即继续循环以检查stop_event
                # app_logger.warning(f"【T1:读帧 {self.stream_id}】预处理队列已满，丢弃当前帧以保持实时性。")
                # 增加短暂休眠，防止在队列持续满时CPU空转
                time.sleep(0.01)
                continue

        # 发送停止信号给下一个线程
        self.preprocess_queue.put(None)
        app_logger.info(f"【T1:读帧 {self.stream_id}】已停止。")

    def _preprocessor_thread(self):
        app_logger.info(f"【T2:预处理 {self.stream_id}】启动。")
        while not self.stop_event.is_set():
            try:
                frame = self.preprocess_queue.get(timeout=0.2)
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
                frame = self.inference_queue.get(timeout=0.2)
                if frame is None:
                    self.postprocess_queue.put(None)
                    break
                
                detection_results = self.det_model.predict(frame).results if self.det_model else []
                self.postprocess_queue.put((frame, detection_results))
            except queue.Empty:
                continue
            except Exception as e:
                app_logger.error(f"【T3:推理-检测 {self.stream_id}】发生错误: {e}", exc_info=True)
        app_logger.info(f"【T3:推理-检测 {self.stream_id}】已停止。")

    def _postprocessor_thread(self):
        app_logger.info(f"【T4:后处理-识别 {self.stream_id}】启动。")
        threshold = self.settings.degirum.recognition_similarity_threshold
        while not self.stop_event.is_set():
            try:
                data = self.postprocess_queue.get(timeout=0.2)
                if data is None:
                    break
                original_frame, detected_faces_data = data
                final_results = []
                if detected_faces_data and self.rec_model:
                    aligned_faces, valid_faces_meta = [], []
                    for face_data in detected_faces_data:
                        landmarks = [lm["landmark"] for lm in face_data.get("landmarks", [])]
                        if len(landmarks) == 5:
                            aligned_face, _ = align_and_crop(original_frame, landmarks)
                            if aligned_face.size > 0:
                                aligned_faces.append(aligned_face)
                                valid_faces_meta.append(face_data)

                    if aligned_faces:
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

                _draw_results_on_frame(original_frame, final_results)
                (flag, encodedImage) = cv2.imencode(".jpg", original_frame)
                if flag:
                    try:
                        self.output_queue.put_nowait(encodedImage.tobytes())
                    except queue.Full:
                        pass
            except queue.Empty:
                continue
            except Exception as e:
                app_logger.error(f"【T4:后处理-识别 {self.stream_id}】发生错误: {e}", exc_info=True)

        try:
            self.output_queue.put_nowait(None)
        except queue.Full:
             pass
        app_logger.info(f"【T4:后处理-识别 {self.stream_id}】已停止。")