# app/core/model_manager.py
import asyncio
from typing import Optional, Tuple
from pathlib import Path
import degirum as dg
import cv2
import numpy as np
import gc
import os

from app.cfg.config import AppSettings, get_app_settings, BASE_DIR
from app.cfg.logging import app_logger
from .process_utils import get_degirum_worker_pids, cleanup_degirum_workers_by_pids

DeGirumModel = dg.model.Model

def create_degirum_model(model_name: str, zoo_url: str) -> DeGirumModel:
    app_logger.info(f"--- 正在加载 DeGirum 模型: '{model_name}' ---")
    app_logger.info(f"  - 模型仓库 (Zoo URL): '{zoo_url}'")
    try:
        model = dg.load_model(
            model_name=model_name,
            inference_host_address=dg.LOCAL,
            zoo_url=zoo_url,
            image_backend='opencv'
        )
        app_logger.info(f"--- ✅ 模型 '{model_name}' 加载成功 ---")
        return model
    except Exception as e:
        app_logger.exception(f"❌ 加载 DeGirum 模型 '{model_name}' 失败: {e}")
        raise RuntimeError(f"加载 DeGirum 模型 '{model_name}' 时出错: {e}") from e

class ModelManager:
    _instance = None
    _detection_model: Optional[DeGirumModel] = None
    _recognition_model: Optional[DeGirumModel] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.settings: AppSettings = get_app_settings()
        return cls._instance

    async def load_models(self):
        if self._detection_model is None or self._recognition_model is None:
            app_logger.info("主进程正在加载 DeGirum 模型...")
            loop = asyncio.get_running_loop()
            self._detection_model = await loop.run_in_executor(None, create_degirum_model, self.settings.degirum.detection_model_name, self.settings.degirum.zoo_url)
            self._recognition_model = await loop.run_in_executor(None, create_degirum_model, self.settings.degirum.recognition_model_name, self.settings.degirum.zoo_url)
            app_logger.info("✅ 主进程所有 DeGirum 模型加载成功。")
            await self._run_startup_self_test()
        else:
            app_logger.info("主进程 DeGirum 模型已加载，跳过重复加载。")

    async def _run_startup_self_test(self):
        app_logger.info("--- 正在执行启动自检 ---")
        test_image_path = BASE_DIR / "app" / "static" / "self_test_face.jpg"
        if not test_image_path.exists():
            app_logger.warning(f"自检失败：未找到测试图片 {test_image_path}。跳过自检。")
            return
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, self._detection_model.predict, str(test_image_path))
            if not result.results:
                app_logger.critical("❌ 【启动自检失败】模型无法在测试图片中检测到任何人脸！")
                raise RuntimeError("DeGirum模型自检失败，服务无法启动。")
            else:
                app_logger.info(f"✅ 【启动自检成功】在测试图片中成功检测到 {len(result.results)} 张人脸。")
        except Exception as e:
            app_logger.exception(f"❌ 自检过程中发生意外错误: {e}")
            raise RuntimeError(f"模型自检过程中出错: {e}") from e

    def get_models(self) -> Tuple[DeGirumModel, DeGirumModel]:
        if self._detection_model is None or self._recognition_model is None:
            raise RuntimeError("DeGirum 模型尚未加载。")
        return self._detection_model, self._recognition_model

    async def release_resources(self):
        app_logger.info("主进程开始执行模型资源强制释放流程...")
        parent_pid = os.getpid()

        app_logger.info("删除主进程中模型对象的Python引用...")
        if self._detection_model:
            del self._detection_model
        if self._recognition_model:
            del self._recognition_model
        self._detection_model = None
        self._recognition_model = None
        app_logger.info("主进程Python引用已清空。")
        
        app_logger.warning("执行全局清理程序，确保杀死所有DeGirum残留的工作进程...")
        pids_to_kill = get_degirum_worker_pids(parent_pid)
        cleanup_degirum_workers_by_pids(pids_to_kill, app_logger)

        app_logger.info("强制执行垃圾回收(GC)...")
        gc.collect()
        app_logger.info("✅ 主进程模型资源强制释放流程已全部完成。")


# 单例实例
model_manager = ModelManager()


async def load_models_on_startup():
    await model_manager.load_models()


async def release_models_on_shutdown():
    await model_manager.release_resources()