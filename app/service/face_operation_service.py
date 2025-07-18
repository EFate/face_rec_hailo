# app/service/face_operation_service.py
from typing import List, Tuple
from pathlib import Path
import numpy as np
import os
from fastapi import HTTPException, status

from app.cfg.config import AppSettings
from app.service.face_dao import FaceDataDAO, LanceDBFaceDataDAO
from app.schema.face_schema import FaceInfo, FaceRecognitionResult, UpdateFaceRequest
from app.cfg.logging import app_logger
# ❗【修改】导入 ModelPool
from app.core.model_manager import ModelPool
from app.core.image_utils import align_and_crop, decode_image, save_face_image

class FaceOperationService:
    """
    【核心修改】通过向模型池借用/归还模型来处理人脸静态业务。
    """
    def __init__(self, settings: AppSettings, model_pool: ModelPool):
        app_logger.info("正在初始化 FaceOperationService (使用模型池)...")
        self.settings = settings
        # ❗【修改】持有对模型池的引用
        self.model_pool = model_pool
        # ... 其他初始化保持不变 ...
        self.face_dao: FaceDataDAO = LanceDBFaceDataDAO(
            db_uri=self.settings.degirum.lancedb_uri,
            table_name=self.settings.degirum.lancedb_table_name,
        )
        self.image_db_path = Path(self.settings.degirum.image_db_path)
        self.image_db_path.mkdir(parents=True, exist_ok=True)

    async def register_face(self, name: str, sn: str, image_bytes: bytes) -> FaceInfo:
        models = None
        try:
            # ❗【修改】从池中获取一套模型
            models = self.model_pool.acquire()
            if models is None:
                raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "服务正忙，请稍后再试。")
            
            detection_model, recognition_model = models
            # ... 后续的检测、识别逻辑完全不变 ...
            img = decode_image(image_bytes)
            detection_result = detection_model.predict(img)
            faces = detection_result.results
            if not faces:
                raise HTTPException(status_code=400, detail="未在图像中检测到任何人脸。")
            if len(faces) > 1:
                raise HTTPException(status_code=400, detail=f"检测到 {len(faces)} 张人脸，注册时必须确保只有一张。")
            face = faces[0]
            landmarks = [lm["landmark"] for lm in face.get("landmarks", [])]
            aligned_face, _ = align_and_crop(img, landmarks)
            recognition_result = recognition_model.predict(aligned_face)
            embedding = recognition_result.results[0]['data'][0]
            x1, y1, x2, y2 = map(int, face["bbox"])
            face_img_to_save = img[y1:y2, x1:x2]
            saved_path = save_face_image(face_img_to_save, sn, self.image_db_path)
            new_record = self.face_dao.create(name, sn, np.array(embedding), saved_path)
            return FaceInfo.model_validate(new_record)
        finally:
            # ❗【修改】确保无论成功或失败，都将模型归还到池中
            if models:
                self.model_pool.release(models)

    async def recognize_face(self, image_bytes: bytes) -> List[FaceRecognitionResult]:
        models = None
        try:
            # ❗【修改】从池中获取一套模型
            models = self.model_pool.acquire()
            if models is None:
                raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "服务正忙，请稍后再试。")

            detection_model, recognition_model = models
            # ... 后续的检测、识别逻辑完全不变 ...
            img = decode_image(image_bytes)
            detection_result = detection_model.predict(img)
            detected_faces_data = detection_result.results
            if not detected_faces_data: return []
            aligned_faces, valid_faces_meta = [], []
            for face_data in detected_faces_data:
                landmarks = [lm["landmark"] for lm in face_data.get("landmarks", [])]
                if len(landmarks) == 5:
                    aligned_face, _ = align_and_crop(img, landmarks)
                    if aligned_face.size > 0:
                        aligned_faces.append(aligned_face)
                        valid_faces_meta.append(face_data)
            if not aligned_faces: return []
            final_results = []
            batch_rec_results = recognition_model.predict_batch(aligned_faces)
            for i, rec_result in enumerate(batch_rec_results):
                embedding = np.array(rec_result.results[0]['data'][0])
                face_meta = valid_faces_meta[i]
                search_res = self.face_dao.search(embedding, self.settings.degirum.recognition_similarity_threshold)
                if search_res:
                    name, sn, similarity = search_res
                    final_results.append(FaceRecognitionResult(
                        name=name, sn=sn, similarity=similarity,
                        box=list(map(int, face_meta["bbox"])),
                        detection_confidence=float(face_meta.get("score", 0.0)),
                        landmark=[lm["landmark"] for lm in face_meta.get("landmarks", [])]
                    ))
            return final_results
        finally:
            # ❗【修改】确保模型被归还
            if models:
                self.model_pool.release(models)

    # 其他纯数据库操作的方法 (get_all_faces, delete_face_by_sn 等) 无需修改
    async def get_all_faces(self) -> List[FaceInfo]:
        all_faces_data = self.face_dao.get_all()
        return [FaceInfo.model_validate(face) for face in all_faces_data]
    async def get_face_by_sn(self, sn: str) -> List[FaceInfo]:
        faces_data = self.face_dao.get_features_by_sn(sn)
        if not faces_data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"未找到SN为 '{sn}' 的人脸记录。")
        return [FaceInfo.model_validate(face) for face in faces_data]
    async def update_face_by_sn(self, sn: str, update_data: UpdateFaceRequest) -> Tuple[int, FaceInfo]:
        update_dict = update_data.model_dump(exclude_unset=True)
        if not update_dict:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="请求体中未提供任何更新数据。")
        await self.get_face_by_sn(sn)
        updated_count = self.face_dao.update_by_sn(sn, update_dict)
        updated_face_info_list = self.face_dao.get_features_by_sn(sn)
        return updated_count, FaceInfo.model_validate(updated_face_info_list[0])
    async def delete_face_by_sn(self, sn: str) -> int:
        records_to_delete = await self.get_face_by_sn(sn)
        deleted_count = self.face_dao.delete_by_sn(sn)
        if deleted_count > 0:
            for record_info in records_to_delete:
                image_path = Path(record_info.image_path)
                if image_path.exists(): os.remove(image_path)
        return deleted_count