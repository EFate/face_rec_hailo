# app/schema/face_schema.py
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, TypeVar, Generic, Dict, Any
from datetime import datetime
import numpy as np

# --- 通用 API 响应模型 ---
T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    """标准API响应格式"""
    code: int = Field(0, description="响应状态码，0表示成功，其他值表示失败。")
    msg: str = Field("Success", description="响应消息。")
    data: Optional[T] = Field(None, description="响应数据。")


# --- 人脸元数据模型 (用于展示和响应) ---
class FaceInfo(BaseModel):
    """单个人脸的详细信息"""
    uuid: str = Field(..., description="人脸特征的唯一ID。")
    name: str = Field(..., description="人脸所属人员的姓名。")
    sn: str = Field(..., description="人脸所属人员的唯一标识SN。")
    registration_time: datetime = Field(..., description="人脸注册时间。")
    image_path: str = Field(..., description="注册图像在文件系统中的路径。")
    extra_info: Optional[Dict[str, Any]] = Field(None, description="预留的额外信息字段。")

    class Config:
        from_attributes = True


# --- API 请求体模型 ---
class UpdateFaceRequest(BaseModel):
    """更新人脸信息请求体"""
    name: Optional[str] = Field(None, description="新的姓名。", example="李四")


# --- API 响应数据体模型 ---
class FaceRegisterResponseData(BaseModel):
    """注册人脸的响应数据"""
    face_info: FaceInfo = Field(..., description="注册成功的人脸信息。")


class FaceRecognitionResult(BaseModel):
    """单次人脸识别结果"""
    name: str = Field(..., description="识别到的人脸姓名。")
    sn: str = Field(..., description="识别到的人脸SN。")
    similarity: float = Field(..., description="与已知人脸特征的余弦相似度，值越大越相似。")
    box: List[int] = Field(..., description="人脸在图像中的边界框 [x1, y1, x2, y2]。")
    detection_confidence: float = Field(..., description="人脸检测置信度。")
    landmark: Optional[List[List[int]]] = Field(None, description="人脸关键点坐标。")

    @field_validator('landmark', mode='before')
    @classmethod
    def landmark_to_list(cls, v):
        if isinstance(v, np.ndarray):
            return v.astype(int).tolist()
        return v

    class Config:
        arbitrary_types_allowed = True


class GetAllFacesResponseData(BaseModel):
    """获取所有人脸列表的响应数据"""
    count: int = Field(..., description="人脸总数。")
    faces: List[FaceInfo] = Field(..., description="已注册人脸的列表。")


class DeleteFaceResponseData(BaseModel):
    """删除人脸的响应数据"""
    sn: str = Field(..., description="被删除的人员SN。")
    deleted_count: int = Field(..., description="成功删除的人脸特征数量。")


class UpdateFaceResponseData(BaseModel):
    """更新人脸信息的响应数据"""
    sn: str = Field(..., description="被更新的人员SN。")
    updated_count: int = Field(..., description="成功更新的人脸特征数量。")
    face_info: FaceInfo = Field(..., description="更新后的人脸信息。")


class HealthCheckResponseData(BaseModel):
    """健康检查响应数据"""
    status: str = Field("ok", description="服务状态。")
    message: str = Field("人脸识别服务正常运行。", description="服务状态信息。")


# --- 视频流管理 Schema ---
class StreamStartRequest(BaseModel):
    """启动视频流请求体"""
    source: str = Field(..., description="视频源。可以是摄像头ID(如 '0') 或 视频文件/URL。", example="0")
    lifetime_minutes: Optional[int] = Field(
        None,
        description="视频流生命周期（分钟）。-1表示永久，不填则使用配置默认值。",
        example=10
    )


class ActiveStreamInfo(BaseModel):
    """单个活动流的基础状态信息（内部使用）"""
    stream_id: str = Field(..., description="流的唯一ID。")
    source: str = Field(..., description="视频源。")
    started_at: datetime = Field(..., description="流启动时间。")
    expires_at: Optional[datetime] = Field(None, description="流过期时间，None表示永不过期。")
    lifetime_minutes: int = Field(..., description="生命周期（分钟），-1表示永久。")

    class Config:
        from_attributes = True


class StreamDetail(ActiveStreamInfo):
    """用于API响应的单个视频流的详细信息"""
    feed_url: str = Field(..., description="用于播放该视频流的完整URL。")


class StopStreamResponseData(BaseModel):
    """停止视频流的响应数据"""
    stream_id: str = Field(..., description="被停止的流ID。")
    message: str = Field("Stream stopped successfully.", description="操作结果信息。")


class GetAllStreamsResponseData(BaseModel):
    """获取所有活动流的响应数据"""
    active_streams_count: int = Field(..., description="当前活动的视频流数量。")
    streams: List[StreamDetail] = Field([], description="所有活动视频流的详细信息列表。")