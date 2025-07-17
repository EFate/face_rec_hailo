# app/core/image_utils.py
import numpy as np
import cv2
import uuid
from typing import List, Tuple, Union
from pathlib import Path
from fastapi import HTTPException

def decode_image(image_bytes: bytes) -> np.ndarray:
    """
    将图像的字节数据解码为OpenCV图像对象。
    """
    try:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("无法解码图像数据，可能是格式不支持或文件损坏。")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"无效的图像文件: {e}")

def save_face_image(face_img: np.ndarray, sn: str, base_path: Path) -> Path:
    """
    将裁剪的人脸图像保存到指定路径。
    """
    file_uuid = str(uuid.uuid4())
    sn_dir = base_path / sn
    sn_dir.mkdir(parents=True, exist_ok=True)
    file_path = sn_dir / f"face_{sn}_{file_uuid}.jpg"
    success = cv2.imwrite(str(file_path), face_img)
    if not success:
        raise HTTPException(status_code=500, detail=f"无法将图像文件写入到路径: {file_path}")
    return file_path


def align_and_crop(img: np.ndarray, landmarks: List[Union[List[float], np.ndarray]], image_size: int = 112) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据给定的关键点对齐并裁剪图像中的人脸。
    此函数直接改编自参考文档，是提升识别精度的核心。

    Args:
        img (np.ndarray): 完整的原始图像 (未经裁剪的边界框)。该图像将被变换。
        landmarks (List): 5个关键点（界标）的列表，格式为 (x, y) 坐标。
                          这些关键点通常包括眼睛、鼻子和嘴巴。
        image_size (int, optional): 图像应被调整到的大小。默认为112。
                                    对于人脸识别模型，通常是112或128。

    Returns:
        Tuple[np.ndarray, np.ndarray]: 对齐后的人脸图像和变换矩阵。
    """
    # ArcFace模型中使用的参考关键点，基于典型的面部界标集。
    _arcface_ref_kps = np.array(
        [
            [38.2946, 51.6963],  # 左眼
            [73.5318, 51.5014],  # 右眼
            [56.0252, 71.7366],  # 鼻子
            [41.5493, 92.3655],  # 左嘴角
            [70.7299, 92.2041],  # 右嘴角
        ],
        dtype=np.float32,
    )

    # 确保输入的界标正好有5个点
    assert len(landmarks) == 5, f"需要5个关键点进行对齐，但收到了 {len(landmarks)} 个。"

    # 验证 image_size 是否为112或128的倍数
    assert image_size % 112 == 0 or image_size % 128 == 0, "图像尺寸必须是112或128的倍数。"

    # 根据所需图像尺寸调整缩放因子（ratio）
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0  # 112缩放无水平偏移
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio  # 128缩放有水平偏移

    # 将缩放和偏移应用于参考关键点
    dst = _arcface_ref_kps * ratio
    dst[:, 0] += diff_x  # 应用水平偏移

    # 估计相似性变换矩阵，以将界标与参考关键点对齐
    M, inliers = cv2.estimateAffinePartial2D(np.array(landmarks, dtype=np.float32), dst, ransacReprojThreshold=1000)
    
    # 健壮性检查：如果对齐失败（例如，关键点质量差），则返回一个空图像
    if inliers is None or not np.all(inliers):
        return np.zeros((image_size, image_size, 3), dtype=np.uint8), np.zeros((2, 3), dtype=np.float32)

    # 应用仿射变换到输入图像以对齐人脸
    aligned_img = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)

    return aligned_img, M