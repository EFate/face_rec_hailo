# app/service/face_dao.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from fastapi import HTTPException, status
from datetime import datetime
import uuid
import lancedb
from lancedb.pydantic import LanceModel, Vector
from pydantic import Field

from app.cfg.logging import app_logger


# LanceFaceSchema 和 FaceDataDAO 保持不变...
class LanceFaceSchema(LanceModel):
    uuid: str = Field(..., description="特征记录的唯一ID")
    vector: Vector(512) = Field(description="512维的人脸特征向量")
    name: str = Field(description="人员姓名")
    sn: str = Field(description="人员唯一标识 (如工号)", default=None)
    image_path: str = Field(description="注册时使用的图片路径")
    registration_time: datetime = Field(description="注册时间", default_factory=datetime.now)


class FaceDataDAO(ABC):
    @abstractmethod
    def create(self, name: str, sn: str, features: np.ndarray, image_path: Path) -> Dict[str, Any]: pass

    @abstractmethod
    def get_all(self) -> List[Dict[str, Any]]: pass

    @abstractmethod
    def get_features_by_sn(self, sn: str) -> List[Dict[str, Any]]: pass

    @abstractmethod
    def delete_by_sn(self, sn: str) -> int: pass

    @abstractmethod
    def update_by_sn(self, sn: str, update_data: Dict[str, Any]) -> int: pass

    @abstractmethod
    def search(self, embedding: np.ndarray, threshold: float, top_k: int = 1) -> Optional[Tuple[str, str, float]]: pass

    @abstractmethod
    def dispose(self): pass


class LanceDBFaceDataDAO(FaceDataDAO):
    def __init__(self, db_uri: str, table_name: str):
        self.db_uri = db_uri
        self.table_name = table_name
        self.db = lancedb.connect(self.db_uri)
        self.table = self._initialize_table()

    def _initialize_table(self) -> lancedb.table.Table:
        try:
            if self.table_name not in self.db.table_names():
                app_logger.info(f"LanceDB 表 '{self.table_name}' 不存在，正在创建...")
                return self.db.create_table(self.table_name, schema=LanceFaceSchema)
            else:
                app_logger.info(f"成功连接到已存在的 LanceDB 表: '{self.table_name}'")
                return self.db.open_table(self.table_name)
        except Exception as e:
            app_logger.error(f"初始化 LanceDB 表失败: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"数据库表初始化失败: {e}")

    # create, get_all, get_features_by_sn, delete_by_sn 方法保持不变...
    def create(self, name: str, sn: str, features: np.ndarray, image_path: Path) -> Dict[str, Any]:
        try:
            new_record = LanceFaceSchema(uuid=str(uuid.uuid4()), vector=features, name=name, sn=sn,
                                         image_path=str(image_path))
            self.table.add([new_record.model_dump()])
            app_logger.info(f"成功向 LanceDB 添加记录: SN={sn}, Name={name}")
            return new_record.model_dump()
        except Exception as e:
            app_logger.error(f"向 LanceDB 添加记录失败: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"数据库写入失败: {e}")

    def get_all(self) -> List[Dict[str, Any]]:
        try:
            return self.table.to_pandas().to_dict('records')
        except Exception as e:
            app_logger.error(f"从 LanceDB 读取所有记录失败: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"数据库读取失败: {e}")

    def get_features_by_sn(self, sn: str) -> List[Dict[str, Any]]:
        try:
            results_df = self.table.search().where(f"sn = '{sn}'").to_pandas()
            return results_df.to_dict('records')
        except Exception as e:
            app_logger.error(f"根据 SN='{sn}' 从 LanceDB 查询失败: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"数据库查询失败: {e}")

    def delete_by_sn(self, sn: str) -> int:
        try:
            initial_count = self.table.count_rows()
            self.table.delete(f"sn = '{sn}'")
            final_count = self.table.count_rows()
            deleted_count = initial_count - final_count
            if deleted_count > 0:
                app_logger.info(f"成功从 LanceDB 中删除 {deleted_count} 条 SN 为 '{sn}' 的记录。")
            return deleted_count
        except Exception as e:
            app_logger.error(f"从 LanceDB 删除 SN='{sn}' 的记录失败: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"数据库删除失败: {e}")

    # --- 【核心优化】重构 update_by_sn 方法 ---
    def update_by_sn(self, sn: str, update_data: Dict[str, Any]) -> int:
        """根据SN更新记录。采用更健壮的“读取-删除-新增”逻辑。"""
        app_logger.warning("LanceDB 的更新操作通过“删除+新增”实现，可能影响性能。")

        # 1. 使用 to_pandas() 一次性读取所有需要的数据
        records_df = self.table.search().where(f"sn = '{sn}'").to_pandas()
        if records_df.empty:
            return 0

        # 2. 删除所有旧记录
        self.delete_by_sn(sn)

        # 3. 遍历DataFrame，手动构建新的记录列表，确保数据完整性
        new_records_to_add = []
        new_name = update_data.get('name')

        for index, row in records_df.iterrows():
            new_record = LanceFaceSchema(
                uuid=str(uuid.uuid4()),  # 必须生成新的UUID
                vector=row['vector'],  # 从DataFrame行中明确获取vector
                name=new_name if new_name else row['name'],  # 应用更新
                sn=row['sn'],
                image_path=row['image_path'],
                registration_time=datetime.now()  # 更新注册时间
            )
            new_records_to_add.append(new_record)

        # 4. 一次性将所有新记录添加回数据库
        try:
            if new_records_to_add:
                self.table.add(new_records_to_add)
            return len(new_records_to_add)
        except Exception as e:
            app_logger.error(f"更新 LanceDB 记录时重新添加数据失败: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"数据库更新的后续写入失败: {e}")

    def search(self, embedding: np.ndarray, threshold: float, top_k: int = 1) -> Optional[Tuple[str, str, float]]:
        try:
            if self.table.count_rows() == 0: return None
            search_result = self.table.search(embedding).metric("cosine").limit(top_k).to_list()
            if not search_result: return None
            best_match = search_result[0]
            similarity = 1 - best_match["_distance"]
            if similarity >= threshold:
                return best_match["name"], best_match["sn"], float(similarity)
            return None
        except Exception:
            return None

    def dispose(self):
        app_logger.info("LanceDB DAO 无需显式资源释放。")
        pass