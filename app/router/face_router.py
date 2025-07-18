# app/router/face_router.py
from typing import List, Optional
from fastapi import (
    APIRouter, Depends, status, File, UploadFile, Form,
    HTTPException, Request, Query, Path as FastApiPath
)
from fastapi.responses import StreamingResponse

from app.schema.face_schema import (
    ApiResponse, FaceRegisterResponseData, FaceRecognitionResult,
    GetAllFacesResponseData, DeleteFaceResponseData, HealthCheckResponseData,
    UpdateFaceRequest, UpdateFaceResponseData, FaceInfo,
    StreamStartRequest, StreamDetail, GetAllStreamsResponseData, StopStreamResponseData
)
# ✅ 导入新的服务类
from app.service.face_operation_service import FaceOperationService
from app.service.stream_manager_service import StreamManagerService

router = APIRouter()

# --- 新的依赖注入函数 ---
def get_face_op_service(request: Request) -> FaceOperationService:
    """依赖注入：获取人脸静态操作服务实例。"""
    return request.app.state.face_op_service

def get_stream_manager_service(request: Request) -> StreamManagerService:
    """依赖注入：获取视频流管理服务实例。"""
    return request.app.state.stream_manager_service

# --- 健康检查 API ---
@router.get(
    "/health",
    response_model=ApiResponse[HealthCheckResponseData],
    summary="健康检查",
    tags=["系统"]
)
async def health_check():
    """检查服务是否正常运行。"""
    return ApiResponse(data=HealthCheckResponseData())


# --- 人脸库管理 API ---
@router.post(
    "/faces",
    response_model=ApiResponse[FaceRegisterResponseData],
    status_code=status.HTTP_201_CREATED,
    summary="注册一张新的人脸",
    tags=["人脸管理"]
)
async def register_face(
    name: str = Form(..., description="人员姓名", example="张三"),
    sn: str = Form(..., description="人员唯一标识 (如工号)", example="EMP001"),
    image_file: UploadFile = File(..., description="上传的人脸图像文件 (jpg, png等)。"),
    face_op_service: FaceOperationService = Depends(get_face_op_service) # ✅ 依赖注入 FaceOperationService
):
    """
    上传一张图片并关联到指定的人员信息。
    - name: 人员姓名 (表单字段)
    - sn: 人员唯一标识 (表单字段)
    - image_file: 包含清晰人脸的图片文件 (文件部分)
    """
    image_bytes = await image_file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="上传的图像文件为空。")

    face_info = await face_op_service.register_face(name, sn, image_bytes)
    return ApiResponse(data=FaceRegisterResponseData(face_info=face_info))


@router.get(
    "/faces",
    response_model=ApiResponse[GetAllFacesResponseData],
    summary="获取所有已注册的人脸信息",
    tags=["人脸管理"]
)
async def get_all_faces(face_op_service: FaceOperationService = Depends(get_face_op_service)): # ✅ 依赖注入
    """获取数据库中所有已注册人脸的元数据列表。"""
    faces = await face_op_service.get_all_faces()
    return ApiResponse(data=GetAllFacesResponseData(count=len(faces), faces=faces))


@router.get(
    "/faces/{sn}",
    response_model=ApiResponse[List[FaceInfo]],
    summary="根据SN获取特定人员的人脸信息",
    tags=["人脸管理"]
)
async def get_face_by_sn(
        sn: str = FastApiPath(..., description="要查询的人员唯一标识SN。", example="EMP001"),
        face_op_service: FaceOperationService = Depends(get_face_op_service) # ✅ 依赖注入
):
    """根据SN获取一个人的所有已注册人脸信息。"""
    faces = await face_op_service.get_face_by_sn(sn)
    return ApiResponse(data=faces)


@router.put(
    "/faces/{sn}",
    response_model=ApiResponse[UpdateFaceResponseData],
    summary="更新指定SN的人员信息",
    tags=["人脸管理"]
)
async def update_face_info(
        sn: str,
        update_data: UpdateFaceRequest,
        face_op_service: FaceOperationService = Depends(get_face_op_service) # ✅ 依赖注入
):
    """
    根据SN更新人员的姓名。
    注意：此接口不用于更换人脸照片，仅用于更新元数据如姓名。
    """
    updated_count, updated_face_info = await face_op_service.update_face_by_sn(sn, update_data)
    return ApiResponse(
        msg=f"成功更新SN为 '{sn}' 的 {updated_count} 条人员信息。",
        data=UpdateFaceResponseData(sn=sn, updated_count=updated_count, face_info=updated_face_info)
    )


@router.delete(
    "/faces/{sn}",
    response_model=ApiResponse[DeleteFaceResponseData],
    summary="删除指定SN的所有人脸记录",
    tags=["人脸管理"]
)
async def delete_face(
        sn: str,
        face_op_service: FaceOperationService = Depends(get_face_op_service) # ✅ 依赖注入
):
    """根据SN删除一个人的所有相关人脸数据和图片文件。"""
    deleted_count = await face_op_service.delete_face_by_sn(sn)
    return ApiResponse(
        msg=f"成功删除SN为 '{sn}' 的 {deleted_count} 条人脸记录。",
        data=DeleteFaceResponseData(sn=sn, deleted_count=deleted_count)
    )


# --- 人脸识别与视频流 API ---
@router.post(
    "/recognize",
    response_model=ApiResponse[List[FaceRecognitionResult]],
    summary="识别静态图像中的人脸",
    tags=["人脸识别"]
)
async def recognize_face(
        image_file: UploadFile = File(..., description="待识别人脸的图像文件。"),
        face_op_service: FaceOperationService = Depends(get_face_op_service) # ✅ 依赖注入
):
    """
    上传一张图片，服务将识别图中的所有人脸，并返回最匹配的已知人员信息。
    """
    image_bytes = await image_file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="上传的图像文件为空。")

    results = await face_op_service.recognize_face(image_bytes)
    if not results:
        return ApiResponse(code=0, msg="在图像中检测到人脸，但未匹配到任何已知身份。", data=[])
    return ApiResponse(data=results)


@router.post(
    "/streams/start",
    response_model=ApiResponse[StreamDetail],
    summary="启动一个视频流任务",
    tags=["视频流管理"]
)
async def start_stream(
        request: Request,
        start_request: StreamStartRequest,
        stream_manager: StreamManagerService = Depends(get_stream_manager_service) # ✅ 依赖注入 StreamManagerService
):
    """
    请求服务器启动一个新的视频流处理任务。
    - **source**: 视频源 (摄像头ID '0', '1', ... 或视频文件路径/URL)
    - **lifetime_minutes**: 流的生命周期（分钟），-1表示永久，不传则使用默认配置。
    """
    stream_info = await stream_manager.start_stream(start_request)
    feed_url = request.url_for('get_stream_feed', stream_id=stream_info.stream_id)
    response_data = StreamDetail(**stream_info.model_dump(), feed_url=str(feed_url))
    return ApiResponse(data=response_data)


@router.get(
    "/streams/feed/{stream_id}",
    summary="获取指定ID的视频流数据",
    tags=["视频流管理"],
    name="get_stream_feed",
    responses={
        200: {"content": {"multipart/x-mixed-replace; boundary=frame": {}}},
        404: {"description": "Stream not found."}
    }
)
async def get_stream_feed(
        stream_id: str,
        stream_manager: StreamManagerService = Depends(get_stream_manager_service) # ✅ 依赖注入
):
    """
    通过此端点获取由 `/streams/start` 启动的视频流。
    此端点专为用在HTML `<img>` 标签的 `src` 属性或类似的流媒体播放器中而设计。
    """
    return StreamingResponse(
        stream_manager.get_stream_feed(stream_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.post(
    "/streams/stop/{stream_id}",
    response_model=ApiResponse[StopStreamResponseData],
    summary="停止一个指定的视频流",
    tags=["视频流管理"]
)
async def stop_stream(
        stream_id: str,
        stream_manager: StreamManagerService = Depends(get_stream_manager_service) # ✅ 依赖注入
):
    """根据 `stream_id` 手动停止一个正在运行的视频流任务。"""
    success = await stream_manager.stop_stream(stream_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Stream with ID '{stream_id}' not found or already stopped.")
    return ApiResponse(data=StopStreamResponseData(stream_id=stream_id))


@router.get(
    "/streams",
    response_model=ApiResponse[GetAllStreamsResponseData],
    summary="获取所有活动的视频流列表",
    tags=["视频流管理"]
)
async def get_all_streams(
        request: Request,
        stream_manager: StreamManagerService = Depends(get_stream_manager_service) # ✅ 依赖注入
):
    """查询并返回当前服务器上所有正在运行的视频流的详细信息列表，包含播放URL。"""
    active_streams_info = await stream_manager.get_all_active_streams_info()

    streams_with_details = [
        StreamDetail(
            **info.model_dump(),
            feed_url=str(request.url_for('get_stream_feed', stream_id=info.stream_id))
        )
        for info in active_streams_info
    ]

    response_data = GetAllStreamsResponseData(
        active_streams_count=len(streams_with_details),
        streams=streams_with_details
    )
    return ApiResponse(data=response_data)