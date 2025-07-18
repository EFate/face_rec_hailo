# app/core/model_manager.py
import gc
import queue
from typing import Tuple, Optional

import degirum as dg

from app.cfg.config import AppSettings
from app.cfg.logging import app_logger
from .process_utils import get_all_degirum_worker_pids, cleanup_degirum_workers_by_pids

DeGirumModel = dg.model.Model

def create_degirum_model(model_name: str, zoo_url: str) -> DeGirumModel:
    """通用模型加载函数，保持不变。"""
    app_logger.info(f"--- 正在加载 DeGirum 模型: '{model_name}' from '{zoo_url}' ---")
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

class ModelPool:
    """
    线程安全的模型池。
    """
    def __init__(self, settings: AppSettings, pool_size: int = 3):
        app_logger.info(f"正在初始化包含 {pool_size} 套模型的【统一模型池】...")
        self.settings = settings
        self.pool_size = pool_size
        self._pool = queue.Queue(maxsize=pool_size)

        try:
            for i in range(pool_size):
                app_logger.info(f"正在加载池中的第 {i+1}/{pool_size} 套模型...")
                detection_model = create_degirum_model(
                    self.settings.degirum.detection_model_name,
                    self.settings.degirum.zoo_url
                )
                recognition_model = create_degirum_model(
                    self.settings.degirum.recognition_model_name,
                    self.settings.degirum.zoo_url
                )
                self._pool.put((detection_model, recognition_model))
            app_logger.info(f"✅ 【统一模型池】初始化成功，当前包含 {self._pool.qsize()} 套可用模型。")
        except Exception as e:
            app_logger.error(f"❌ 初始化模型池失败: {e}")
            raise

    def acquire(self, timeout: float = 0.1) -> Optional[Tuple[DeGirumModel, DeGirumModel]]:
        """从池中获取一套模型。"""
        try:
            app_logger.debug(f"尝试从模型池中获取模型 (可用: {self._pool.qsize()}/{self.pool_size})...")
            models = self._pool.get(timeout=timeout)
            app_logger.debug("成功获取到一套模型。")
            return models
        except queue.Empty:
            app_logger.warning(
                f"模型池资源不足！在 {timeout} 秒内无法获取到可用模型。"
            )
            return None

    def release(self, models: Tuple[DeGirumModel, DeGirumModel]):
        """将一套模型归还到池中。"""
        app_logger.debug("将一套模型归还到模型池...")
        self._pool.put(models)
        app_logger.debug(f"归还成功 (可用: {self._pool.qsize()}/{self.pool_size})。")

    def dispose(self):
        """
        【核心修复】应用关闭时，强制清理所有残留工作进程，然后释放模型。
        修复了因等待模型正常释放超时而导致的程序崩溃问题。
        """
        app_logger.warning("正在执行【统一模型池】资源释放程序...")

        # 1. 【首要步骤】强制杀死所有 DeGirum 工作进程。
        # 这模拟了 `ps | grep | xargs kill` 的行为，避免了后续操作的超时。
        app_logger.warning("执行全局清理：立即终止所有DeGirum残留的工作进程...")
        try:
            pids_to_kill = get_all_degirum_worker_pids()
            cleanup_degirum_workers_by_pids(pids_to_kill, app_logger)
        except Exception as e:
            app_logger.error(f"【严重】在执行进程清理时发生意外错误: {e}", exc_info=True)


        # 2. 清空队列并尝试释放Python模型对象。
        app_logger.warning("正在清空模型队列并释放Python侧的模型对象...")
        while not self._pool.empty():
            try:
                det_model, rec_model = self._pool.get_nowait()
                # 即使这里因为工作进程被杀而报错，我们也捕获它并继续。
                del det_model
                del rec_model
            except queue.Empty:
                break
            except Exception as e:
                # 捕获因工作进程已死而导致的通信错误，这是预期的。
                app_logger.warning(f"释放模型对象时捕获到一个预期中的错误（因为工作进程已被终止）：{e}")
        
        gc.collect()
        app_logger.info("✅ 【统一模型池】已清空，相关硬件资源已强制释放。")