import sys
import logging
from loguru import logger
from typing import Any
from app.cfg.config import AppSettings # 导入 AppSettings 类型提示


# Loguru 的全局日志器实例
# 应用程序中所有的日志记录都应通过此 app_logger 实例进行，
# 以确保日志格式和输出目标（控制台、文件）的一致性。
app_logger = logger


class LoguruInterceptHandler(logging.Handler):
    """
    一个标准库 `logging` 的处理程序，用于将 `logging` 模块发出的日志记录
    重定向到 Loguru 日志系统。这确保了所有日志（包括来自第三方库的日志）
    都由 Loguru 统一管理和格式化。
    """

    def emit(self, record: logging.LogRecord) -> None:
        """
        处理传入的日志记录。
        """
        try:
            # 尝试将标准库的日志级别名称映射到 Loguru 的级别名称
            level = app_logger.level(record.levelname).name
        except ValueError:
            # 如果映射失败，则使用原始的级别号
            level = record.levelno

        # 获取调用栈信息，用于 Loguru 的 `backtrace` 和 `diagnose` 功能
        # 确保 Loguru 报告的源文件和行号是实际日志调用的位置，而不是这个处理程序内部
        frame, depth = logging.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        # 使用 Loguru 记录日志，并传递原始的异常信息
        app_logger.opt(
            depth=depth,
            exception=record.exc_info
        ).log(level, record.getMessage())


def setup_logging(settings: AppSettings) -> None:
    """
    初始化 Loguru 日志系统。
    根据传入的 AppSettings 配置，设置控制台和文件日志的输出格式、级别、
    文件轮转策略等。
    此函数应在应用程序启动时，且配置加载完成后调用一次。
    """
    # 移除 Loguru 默认添加的控制台处理器，避免重复配置
    app_logger.remove()

    log_level = settings.logging.level.upper() # 从配置获取日志级别
    is_debug = settings.app.debug # 从配置获取调试模式状态

    # 配置控制台日志输出
    app_logger.add(
        sys.stderr,  # 日志输出到标准错误流
        level=log_level,  # 日志级别
        format=(
            "<level>{level.icon}</level> "  # 级别图标
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "  # 时间戳
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "  # 模块名、函数名、行号
            "<level>{message}</level>"  # 日志消息
        ),
        colorize=True,  # 启用彩色输出
        backtrace=is_debug,  # 调试模式下显示追溯
        diagnose=is_debug,  # 调试模式下显示变量诊断信息
        enqueue=True, # 启用异步写入，提高性能
    )

    # 配置文件日志输出
    app_logger.add(
        settings.logging.file_path,  # 日志文件路径
        level=log_level,  # 日志级别
        rotation=f"{settings.logging.max_bytes} B",  # 文件轮转策略：按大小
        retention=settings.logging.backup_count,  # 备份文件数量
        compression="zip",  # 历史日志文件压缩格式
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        backtrace=is_debug,
        diagnose=is_debug,
        enqueue=True, # 启用异步写入
    )

    # 配置标准库 `logging` 的根记录器，使其日志通过 LoguruInterceptHandler 转发
    # `force=True` 确保即使已经有处理器，也会被替换
    logging.basicConfig(handlers=[LoguruInterceptHandler()], level=0, force=True)

    # 禁用所有上层 `logging` 记录器的传播，以避免日志重复
    # 遍历所有已知的 logger，并将其 `propagate` 属性设置为 False
    for name in logging.root.manager.loggerDict:
        logging.getLogger(name).handlers = [] # 清空处理器
        logging.getLogger(name).propagate = False # 禁用传播