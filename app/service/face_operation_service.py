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
from app.core.model_manager import ModelManager
# ✅ 导入新的通用工具函数
from app.core.image_utils import align_and_crop, decode_image, save_face_image

class FaceOperationService:
    """
    负责处理核心人脸业务，如注册、识别、数据库管理等（静态图片操作）。
    """
    def __init__(self, settings: AppSettings, model_manager: ModelManager):
        app_logger.info("正在初始化 FaceOperationService...")
        self.settings = settings
        self.model_manager = model_manager
        self.detection_model, self.recognition_model = self.model_manager.get_models()
        self.face_dao: FaceDataDAO = LanceDBFaceDataDAO(
            db_uri=self.settings.degirum.lancedb_uri,
            table_name=self.settings.degirum.lancedb_table_name,
        )
        self.image_db_path = Path(self.settings.degirum.image_db_path)
        self.image_db_path.mkdir(parents=True, exist_ok=True)

    async def initialize(self):
        """异步初始化过程（如果需要）。"""
        app_logger.info("✅ FaceOperationService 初始化完毕。")
        # 此处可以放置需要异步执行的初始化代码
        pass

    async def register_face(self, name: str, sn: str, image_bytes: bytes) -> FaceInfo:
        img = decode_image(image_bytes)
        detection_result = self.detection_model.predict(img)
        faces = detection_result.results
        if not faces:
            raise HTTPException(status_code=400, detail="未在图像中检测到任何人脸。")
        if len(faces) > 1:
            raise HTTPException(status_code=400, detail=f"检测到 {len(faces)} 张人脸，注册时必须确保只有一张。")

        face = faces[0]
        det_score = face.get("score", 0.0)
        if det_score < self.settings.degirum.recognition_det_score_threshold:
            raise HTTPException(status_code=400, detail=f"人脸质量不佳，检测置信度({det_score:.2f})过低。")

        x1, y1, x2, y2 = map(int, face["bbox"])
        face_img_to_save = img[y1:y2, x1:x2]
        if face_img_to_save.size == 0:
            raise HTTPException(status_code=400, detail="根据边界框裁剪出的人脸图像为空。")

        landmarks = [lm["landmark"] for lm in face.get("landmarks", [])]
        if len(landmarks) != 5:
            raise HTTPException(status_code=400, detail=f"人脸对齐失败：需要5个关键点，但检测到 {len(landmarks)} 个。")

        aligned_face, _ = align_and_crop(img, landmarks)
        if aligned_face.size == 0:
            raise HTTPException(status_code=400, detail="人脸对齐后图像为空，对齐可能失败。")

        recognition_result = self.recognition_model.predict(aligned_face)
        embedding = recognition_result.results[0]['data'][0]
        saved_path = save_face_image(face_img_to_save, sn, self.image_db_path)
        new_record = self.face_dao.create(name, sn, np.array(embedding), saved_path)
        return FaceInfo.model_validate(new_record)

    async def recognize_face(self, image_bytes: bytes) -> List[FaceRecognitionResult]:
        img = decode_image(image_bytes)
        detection_result = self.detection_model.predict(img)
        detected_faces_data = detection_result.results
        if not detected_faces_data:
            return []

        aligned_faces = []
        valid_faces_meta = []
        for face_data in detected_faces_data:
            landmarks = [lm["landmark"] for lm in face_data.get("landmarks", [])]
            if len(landmarks) == 5:
                aligned_face, _ = align_and_crop(img, landmarks)
                if aligned_face.size > 0:
                    aligned_faces.append(aligned_face)
                    valid_faces_meta.append(face_data)
                else:
                    app_logger.warning("跳过一张人脸，因对齐后图像为空。")
            else:
                app_logger.warning(f"跳过一张人脸，因对齐需要5个关键点，但检测到 {len(landmarks)} 个。")

        if not aligned_faces:
            return []

        final_results = []
        batch_rec_results = self.recognition_model.predict_batch(aligned_faces)
        for i, rec_result in enumerate(batch_rec_results):
            embedding = np.array(rec_result.results[0]['data'][0])
            face_meta = valid_faces_meta[i]
            search_res = self.face_dao.search(embedding, self.settings.degirum.recognition_similarity_threshold)

            if search_res:
                name, sn, similarity = search_res
                face_landmarks = face_meta.get("landmarks", [])
                landmark_coords = [lm["landmark"] for lm in face_landmarks] if face_landmarks else None
                final_results.append(FaceRecognitionResult(
                    name=name, sn=sn, similarity=similarity,
                    box=list(map(int, face_meta["bbox"])),
                    detection_confidence=float(face_meta.get("score", 0.0)),
                    landmark=landmark_coords
                ))
        return final_results

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

        # 验证记录是否存在
        await self.get_face_by_sn(sn)

        updated_count = self.face_dao.update_by_sn(sn, update_dict)
        if updated_count == 0:
            app_logger.warning(f"更新操作成功，但SN为'{sn}'的0条记录被更新。")

        updated_face_info_list = self.face_dao.get_features_by_sn(sn)
        if not updated_face_info_list:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="更新后无法找回记录，数据可能不一致。")

        app_logger.info(f"人员信息已更新: SN={sn}, 新数据={update_dict}, 影响记录数={updated_count}")
        return updated_count, FaceInfo.model_validate(updated_face_info_list[0])

    async def delete_face_by_sn(self, sn: str) -> int:
        # 使用get_face_by_sn来隐式检查记录是否存在，如果不存在会抛出404异常
        records_to_delete = await self.get_face_by_sn(sn)

        deleted_count = self.face_dao.delete_by_sn(sn)
        if deleted_count > 0:
            for record_info_dict in records_to_delete:
                try:
                    # 从模型转换回来的字典中安全地获取image_path
                    image_path_str = record_info_dict.image_path
                    if image_path_str:
                        image_path = Path(image_path_str)
                        if image_path.exists():
                            os.remove(image_path)
                            app_logger.info(f"已删除关联图片文件: {image_path}")
                except Exception as e:
                    app_logger.error(f"删除图片文件 {record_info_dict.image_path} 失败: {e}")
        return deleted_count