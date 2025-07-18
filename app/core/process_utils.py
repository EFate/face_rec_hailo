# app/core/process_utils.py
import os
import psutil
import signal
from typing import Set
from logging import Logger

def get_all_degirum_worker_pids() -> Set[int]:
    """
    【优化】获取当前系统上所有正在运行的DeGirum工作进程的PID集合。
    此函数通过扫描所有进程的命令行来识别目标进程，确保全面清理。
    """
    worker_pids = set()
    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline')
            # DeGirum的工作进程通常通过执行 pproc_worker.py 脚本启动
            if cmdline and any("degirum/pproc_worker.py" in s for s in cmdline):
                worker_pids.add(proc.info['pid'])
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            # 进程可能在我们检查时已经消失、无权访问或是僵尸进程，直接跳过
            continue
    return worker_pids

def cleanup_degirum_workers_by_pids(pids_to_kill: Set[int], logger: Logger):
    """
    【优化】根据提供的PID集合，强制终止DeGirum工作进程。
    使用 SIGKILL 信号确保进程被立即终止，以释放硬件资源。
    """
    if not pids_to_kill:
        logger.info("【进程清理】没有检测到需要清理的DeGirum工作进程。")
        return

    logger.warning(f"【进程清理】将要强制终止PID为 {list(pids_to_kill)} 的DeGirum工作进程...")
    killed_count = 0
    for pid in pids_to_kill:
        try:
            # 使用SIGKILL信号强制、立即终止进程
            # 在Linux/macOS上，这是最可靠的强制退出方式
            os.kill(pid, signal.SIGKILL)
            logger.info(f"【进程清理】已向PID {pid} 发送SIGKILL信号。")
            killed_count += 1
        except ProcessLookupError:
            logger.warning(f"【进程清理】尝试终止PID {pid} 时失败，进程已不存在。")
        except Exception as e:
            logger.error(f"【进程清理】终止PID {pid} 时发生未知错误: {e}")

    if killed_count > 0:
        logger.info(f"【进程清理】成功终止了 {killed_count} 个DeGirum工作进程。")