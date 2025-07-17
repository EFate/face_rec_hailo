import cv2
import time
from datetime import datetime

def check_rtsp_stream(url, timeout=10):
    """检测RTSP流是否正常运行"""
    print(f"开始检测RTSP流: {url}")
    print(f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 初始化变量
    success = False
    frame = None
    error_message = None
    
    try:
        # 尝试连接RTSP流
        start_time = time.time()
        cap = cv2.VideoCapture(url)
        
        # 设置超时时间
        while (time.time() - start_time) < timeout:
            ret, frame = cap.read()
            if ret:
                success = True
                height, width = frame.shape[:2]
                fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"成功获取视频帧，分辨率: {width}x{height}, FPS: {fps:.2f}")
                break
            else:
                print("尝试获取视频帧失败，重试中...")
                time.sleep(0.5)
        
        # 释放资源
        cap.release()
        
        if not success:
            error_message = f"连接超时，无法获取视频帧（超时时间: {timeout}秒）"
    
    except cv2.error as e:
        error_message = f"OpenCV错误: {str(e)}"
    except Exception as e:
        error_message = f"发生未知错误: {str(e)}"
    
    # 返回结果
    return {
        "success": success,
        "error_message": error_message,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

if __name__ == "__main__":
    # RTSP地址
    rtsp_url = "rtsp://172.16.4.152:8554/test"
    
    # 执行检测
    result = check_rtsp_stream(rtsp_url)
    
    # 输出结果
    if result["success"]:
        print("RTSP流检测结果: ✅ 正常运行")
    else:
        print(f"RTSP流检测结果: ❌ 无法连接")
        print(f"错误信息: {result['error_message']}")    