# app/cfg/config.py
import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from functools import lru_cache
from pydantic import BaseModel, Field, BeforeValidator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Annotated


# --- 路径定义 ---
def get_base_dir() -> Path:
    """计算项目根目录。"""
    return Path(__file__).resolve().parent.parent.parent


BASE_DIR = get_base_dir()
ENV_FILE = BASE_DIR / ".env"
LOGS_DIR = BASE_DIR / "logs"
CONFIG_DIR = BASE_DIR / "app" / "cfg"
DATA_DIR = BASE_DIR / "data"
# 定义模型仓库（Zoo）的路径
MODEL_ZOO_DIR = BASE_DIR / "data" / "zoo"
LANCEDB_DATA_DIR = BASE_DIR / "data" / "lancedb"

# --- 自定义类型 ---
FilePath = Annotated[Path, BeforeValidator(lambda v: Path(v) if isinstance(v, str) else v)]


# --- 配置模型定义 ---
class AppConfig(BaseModel):
    title: str = Field("高性能人脸识别服务 (DeGirum版)", description="应用程序名称。")
    description: str = Field("基于FastAPI、DeGirum和LanceDB构建", description="应用程序描述。")
    version: str = Field("6.0.0-Pipeline-Batch", description="应用程序版本。")
    debug: bool = Field(False, description="是否开启调试模式。")
    max_concurrent_tasks: int = Field(3, description="系统允许的最大并发AI任务数（模型池大小）。")
    stream_default_lifetime_minutes: int = Field(10, description="视频流默认生命周期（分钟），-1表示永久。")
    stream_cleanup_interval_seconds: int = Field(60, description="清理过期视频流的后台任务运行间隔（秒）。")
    font_path: FilePath = Field(
        BASE_DIR / "app" / "static" / "fonts" / "wqy-microhei.ttc",
        description="用于在图片上绘制中文的字体文件路径。"
    )


class ServerConfig(BaseModel):
    host: str = Field("0.0.0.0", description="服务器监听地址。")
    port: int = Field(8000, description="服务器监听端口。")
    reload: bool = Field(False, description="是否开启热重载（仅开发环境）。")


class LoggingConfig(BaseModel):
    level: str = Field("INFO", description="日志级别。")
    file_path: FilePath = Field(LOGS_DIR / "app.log", description="日志文件绝对路径。")
    max_bytes: int = Field(10 * 1024 * 1024, description="单个日志文件最大字节数（10MB）。")
    backup_count: int = Field(5, description="日志文件备份数量。")

    def model_post_init__(self, __context: Any) -> None:
        if self.file_path: self.file_path.parent.mkdir(parents=True, exist_ok=True)



class DeGirumConfig(BaseModel):
    # 模型仓库（Zoo）的URL，这里使用本地文件系统
    zoo_url: str = Field(f"file://{MODEL_ZOO_DIR.absolute()}", description="DeGirum 模型仓库的URL。")
    # 人脸检测模型名称
    detection_model_name: str = Field("scrfd_10g--640x640_quant_hailort_hailo8_1", description="人脸检测模型名。")
    # 人脸识别（特征提取）模型名称
    recognition_model_name: str = Field("arcface_mobilefacenet--112x112_quant_hailort_hailo8_1", description="人脸识别模型名。")
    # 人脸识别余弦相似度阈值
    recognition_similarity_threshold: float = Field(0.5, description="人脸识别余弦相似度阈值。")
    # 人脸检测的最低置信度分数
    recognition_det_score_threshold: float = Field(0.5, description="人脸检测的最低置信度分数。")
    # 存储注册人脸图像的根目录
    image_db_path: FilePath = Field(DATA_DIR / "faces", description="用于存储注册人脸图像的根目录。")
    # LanceDB 数据库文件的存储目录
    lancedb_uri: str = Field(str(LANCEDB_DATA_DIR), description="LanceDB 数据库文件的存储目录。")
    # 用于存储人脸特征的表名
    lancedb_table_name: str = Field("faces_v4_degirum_pipeline", description="用于存储人脸特征的表名。")


# --- 主配置类 ---
class AppSettings(BaseSettings):
    app: AppConfig = Field(default_factory=AppConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    degirum: DeGirumConfig = Field(default_factory=DeGirumConfig) # ✅ 使用新的配置模型

    model_config = SettingsConfigDict(
        env_file=ENV_FILE, env_file_encoding="utf-8", case_sensitive=False,
        extra="ignore", populate_by_name=True, env_nested_delimiter="__",
    )


# --- 配置加载器 (保持不变) ---
class ConfigLoader:
    @staticmethod
    def load_yaml_configs(env: Optional[str] = None) -> Dict[str, Any]:
        current_env = env or os.getenv("APP_ENV", "development").lower()
        config: Dict[str, Any] = {}
        default_path = CONFIG_DIR / "default.yaml"
        if default_path.exists():
            try:
                with open(default_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}
            except Exception as e:
                print(f"警告：加载默认配置文件 {default_path} 失败: {e}")
        env_path = CONFIG_DIR / f"{current_env}.yaml"
        if env_path.exists():
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    env_config = yaml.safe_load(f) or {}
                    config = ConfigLoader._deep_merge_dicts(config, env_config)
            except Exception as e:
                print(f"警告：加载环境特定配置文件 {env_path} 失败: {e}")
        return config

    @staticmethod
    def _deep_merge_dicts(base: Dict, updates: Dict) -> Dict:
        merged = base.copy()
        for key, value in updates.items():
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                merged[key] = ConfigLoader._deep_merge_dicts(merged[key], value)
            else:
                merged[key] = value
        return merged


@lru_cache(maxsize=1)
def get_app_settings(env_override: Optional[str] = None) -> AppSettings:
    current_env = env_override or os.getenv("APP_ENV", "development")
    yaml_data = ConfigLoader.load_yaml_configs(current_env)
    base_settings = AppSettings.model_validate(yaml_data)
    env_aware_settings = AppSettings()
    env_overrides = env_aware_settings.model_dump(exclude_unset=True)
    final_data = ConfigLoader._deep_merge_dicts(base_settings.model_dump(), env_overrides)
    final_settings = AppSettings.model_validate(final_data)
    Path(final_settings.degirum.lancedb_uri).mkdir(parents=True, exist_ok=True)
    return final_settings