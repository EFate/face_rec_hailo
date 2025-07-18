# app/main.py
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.exceptions import HTTPException
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.cfg.config import DATA_DIR

from app.cfg.config import get_app_settings
from app.cfg.logging import app_logger

from app.core.model_manager import ModelPool
from app.router.face_router import router as face_router

from app.service.face_operation_service import FaceOperationService
from app.service.stream_manager_service import StreamManagerService
from app.schema.face_schema import ApiResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    【优化】应用生命周期管理器，负责初始化和销毁模型池，并确保所有资源有序关闭。
    """
    # --- 启动任务 ---
    app_logger.info("============== 应用程序启动 ==============")
    settings = get_app_settings()
    app.state.settings = settings

    # 1. ❗ 初始化统一模型池，大小设置为2
    app_logger.info("--> 正在初始化模型池...")
    model_pool = ModelPool(settings=settings, pool_size=settings.app.max_concurrent_tasks)
    app.state.model_pool = model_pool
    app_logger.info("✅ 统一模型池初始化完成。")

    # 2. ❗ 初始化服务，并将模型池注入
    app_logger.info("--> 正在初始化服务...")
    face_op_service = FaceOperationService(settings=settings, model_pool=model_pool)
    app.state.face_op_service = face_op_service

    stream_manager_service = StreamManagerService(settings=settings, model_pool=model_pool)
    app.state.stream_manager_service = stream_manager_service
    app_logger.info("✅ 所有服务初始化完成。")

    # 3. 启动后台任务 (保持不变)
    app_logger.info("--> 正在启动后台任务...")
    cleanup_task = asyncio.create_task(stream_manager_service.cleanup_expired_streams())
    app.state.cleanup_task = cleanup_task
    app_logger.info("✅ 启动了周期性清理过期视频流的后台任务。")

    app_logger.info("🎉============== 应用程序准备就绪 ==============🎉")
    yield
    # --- 关闭任务 ---
    app_logger.info("============== 应用程序正在关闭 ==============")

    # 1. 停止后台任务
    app_logger.info("--> 正在停止后台任务...")
    if hasattr(app.state, 'cleanup_task') and not app.state.cleanup_task.done():
        app.state.cleanup_task.cancel()
        try:
            await app.state.cleanup_task
        except asyncio.CancelledError:
            app_logger.info("✅ 视频流清理任务已取消。")

    # 2. 【优化】异步关闭所有视频流
    app_logger.info("--> 正在停止所有活动视频流...")
    if hasattr(app.state, 'stream_manager_service'):
        await app.state.stream_manager_service.stop_all_streams()
        app_logger.info("✅ 所有活动视频流已停止。")

    # 3. ❗ 释放模型池中的所有资源（这将触发进程清理）
    app_logger.info("--> 正在释放模型池并执行最终清理...")
    if hasattr(app.state, 'model_pool'):
        app.state.model_pool.dispose()
    app_logger.info("✅ 模型池已释放。")

    app_logger.info("✅==============所有清理任务完成，再见==============✅")

def create_app() -> FastAPI:
    # 此函数内容基本不变
    app_settings = get_app_settings()
    app = FastAPI(
        lifespan=lifespan,
        title=app_settings.app.title,
        description=app_settings.app.description,
        version=app_settings.app.version,
        docs_url=None,
        redoc_url=None,
    )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(status_code=exc.status_code,
                            content=ApiResponse(code=exc.status_code, msg=exc.detail).model_dump())
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        app_logger.exception(f"未处理的服务器内部错误: {exc}")
        return JSONResponse(status_code=500, content=ApiResponse(code=500, msg="服务器内部错误").model_dump())
    app.include_router(face_router, prefix="/api/face", tags=["人脸服务"])
    
    # 挂载静态文件和数据目录
    STATIC_FILES_DIR = Path("app/static")
    if STATIC_FILES_DIR.exists(): app.mount("/static", StaticFiles(directory=STATIC_FILES_DIR), name="static")
    if DATA_DIR.exists(): app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")

    # 自定义Swagger UI
    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        return get_swagger_ui_html(openapi_url=app.openapi_url, title=app.title + " - API Docs",
                                   swagger_js_url="/static/swagger-ui/swagger-ui-bundle.js",
                                   swagger_css_url="/static/swagger-ui/swagger-ui.css")
    
    return app

app = create_app()