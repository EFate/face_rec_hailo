# run.py
import os
import socket
from pathlib import Path
from typing import Annotated, Optional

import typer
import uvicorn

from app.cfg.config import get_app_settings, AppSettings
from app.cfg.logging import app_logger as logger, setup_logging

# 创建 Typer 应用实例
# 移除了错误的 context_obj 参数
app = typer.Typer(
    pretty_exceptions_enable=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def init_app_state(env: Optional[str] = None) -> AppSettings:
    """
    初始化应用状态：设置环境变量，加载配置，并配置日志。
    """
    # 如果通过命令行 --env 传入了值，则优先使用它来设置环境变量
    if env:
        os.environ["APP_ENV"] = env

    # 清除lru_cache，确保配置是基于当前环境加载的
    get_app_settings.cache_clear()

    # 加载配置，get_app_settings 会自动处理所有加载逻辑
    current_settings = get_app_settings()

    # 使用加载好的配置来设置日志系统
    setup_logging(current_settings)

    logger.info(f"⚙️  应用环境已确立: {os.getenv('APP_ENV', 'development').upper()}")
    return current_settings


def get_local_ip() -> str:
    """获取本机IP地址，用于在日志中提供可访问的URL。"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def _print_config_details(settings_instance: AppSettings):
    """打印详细配置信息（敏感信息脱敏）。"""
    logger.info("\n--- 📦 应用配置 (AppConfig) ---")
    logger.info(f"  - 标题: {settings_instance.app.title}")
    logger.info(f"  - 版本: {settings_instance.app.version}")
    logger.info(f"  - 调试模式: {'✅ 开启' if settings_instance.app.debug else '❌ 关闭'}")

    logger.info("\n--- ⚙️ 服务器配置 (ServerConfig) ---")
    logger.info(f"  - 主机: {settings_instance.server.host}")
    logger.info(f"  - 端口: {settings_instance.server.port}")
    logger.info(f"  - 热重载: {'✅ 开启' if settings_instance.server.reload else '❌ 关闭'}")

    logger.info("\n--- 📂 日志配置 (LoggingConfig) ---")
    logger.info(f"  - 级别: {settings_instance.logging.level.upper()}")
    logger.info(f"  - 路径: {settings_instance.logging.file_path.resolve()}")

    logger.info("\n--- 🗄️ 数据库配置 (DatabaseConfig) ---")
    logger.info(f"  - 类型: {settings_instance.insightface.storage_type.value}")
    logger.info(f"  - URL: {settings_instance.database.url}")
    logger.info(f"  - SQL打印: {'✅ 开启' if settings_instance.database.echo else '❌ 关闭'}")

    logger.info("\n--- 🧠 AI模型配置 (InsightFaceConfig) ---")
    logger.info(f"  - 模型包: {settings_instance.insightface.model_pack_name}")
    logger.info(f"  - 执行设备: {settings_instance.insightface.providers}")
    logger.info(f"  - 识别阈值: {settings_instance.insightface.recognition_threshold}")

    logger.info("\n--- 🔐 安全配置 (SecurityConfig) ---")
    secret_key = settings_instance.security.secret_key
    masked_key = secret_key[:4] + "****" + secret_key[-4:] if len(secret_key) > 8 else "****"
    logger.info(f"  - 密钥: {masked_key}")


@app.callback(invoke_without_command=True)
def main(
        ctx: typer.Context,
        env: Annotated[
            Optional[str],
            typer.Option("--env", "-e", help="指定运行环境 (development, production)。", envvar="APP_ENV",
                         show_envvar=True),
        ] = None,
        show_config: Annotated[
            bool, typer.Option("--show-config", help="显示当前环境的详细配置并退出。")
        ] = False,
        version: Annotated[
            bool, typer.Option("--version", "-v", help="显示应用版本信息并退出。")
        ] = False,
):
    """
    高性能人脸识别服务 FastAPI - 命令行接口。
    此函数作为入口，负责加载配置并将其存储在 ctx.obj 中，供所有子命令使用。
    """
    # 初始化应用状态并获取配置
    settings = init_app_state(env)
    # 将加载好的配置实例存储在 Typer 的上下文对象中
    # ctx.obj 默认存在，可以直接赋值
    ctx.obj = settings

    if version:
        typer.echo(f"{settings.app.title} - Version: {settings.app.version}")
        raise typer.Exit()

    if show_config:
        _print_config_details(settings)
        raise typer.Exit()

    if ctx.invoked_subcommand is None:
        typer.echo("没有指定子命令。使用 'start' 来启动服务。")
        typer.echo("使用 '--help' 查看可用命令。")


@app.command(name="start")
def start_server(
        ctx: typer.Context,
        host: Annotated[
            Optional[str],
            typer.Option("--host", help="覆盖配置文件中的服务器主机。优先级最高。")
        ] = None,
        port: Annotated[
            Optional[int],
            typer.Option("--port", help="覆盖配置文件中的服务器端口。优先级最高。")
        ] = None,
):
    """
    启动 FastAPI Uvicorn 服务器。
    """
    # 从 Typer 上下文中获取已加载的配置
    settings: AppSettings = ctx.obj

    # 命令行参数拥有最高优先级，覆盖所有配置文件和环境变量
    final_host = host if host is not None else settings.server.host
    final_port = port if port is not None else settings.server.port
    final_reload = settings.server.reload

    logger.info(f"\n🚀 准备启动服务器: {settings.app.title} v{settings.app.version}")
    logger.info(f"  - 最终监听地址: http://{final_host}:{final_port}")
    if final_host == "0.0.0.0":
        local_ip = get_local_ip()
        logger.info(f"  - 本机可访问: http://127.0.0.1:{final_port}")
        logger.info(f"  - 局域网可访问: http://{local_ip}:{final_port}")
    logger.info(f"  - 热重载模式: {'✅ 开启' if final_reload else '❌ 关闭'}")

    try:
        uvicorn.run(
            "app.main:app",
            host=final_host,
            port=final_port,
            reload=final_reload,
            log_level=settings.logging.level.lower(),
            log_config=None  # 由 Loguru 完全接管
        )
    except Exception as e:
        logger.critical(f"⚠️ Uvicorn 服务器启动失败: {e}", exc_info=True)
        raise typer.Exit(code=1)

# 【核心修正】导入 multiprocessing 并设置启动方式
import multiprocessing as mp
if __name__ == "__main__":
    # 在应用启动的最初阶段，强制将多进程启动方式设置为 'spawn'。
    # 避免在Linux上由 'fork' 模式引发的CUDA上下文冲突或C扩展库线程不安全等问题。
    try:
        mp.set_start_method('spawn', force=True)
        logger.info("✅ Multiprocessing start method has been set to 'spawn'.")
    except RuntimeError:
        # 如果上下文已经设置，会抛出 RuntimeError，这在某些情况下是正常的，可以忽略。
        pass

    app()