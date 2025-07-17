# app/core/process_utils.py
import os
import psutil
import signal
from typing import Set
from logging import Logger

def get_degirum_worker_pids(parent_pid: int) -> Set[int]:
    """
    获取指定父进程下的所有DeGirum工作进程的PID集合。
    """
    worker_pids = set()
    for proc in psutil.process_iter(['pid', 'ppid', 'cmdline']):
        try:
            if proc.info['ppid'] == parent_pid:
                cmdline = proc.info.get('cmdline')
                if cmdline and any("degirum/pproc_worker.py" in s for s in cmdline):
                    worker_pids.add(proc.info['pid'])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            # 进程可能在我们检查时已经消失，或者我们没有权限访问
            continue
    return worker_pids

def cleanup_degirum_workers_by_pids(pids_to_kill: Set[int], logger: Logger):
    """
    根据提供的PID集合，强制终止DeGirum工作进程。
    """
    if not pids_to_kill:
        logger.info("【进程清理】没有需要清理的目标PID。")
        return

    logger.warning(f"【进程清理】将要强制终止PID为 {pids_to_kill} 的DeGirum工作进程...")
    killed_count = 0
    for pid in pids_to_kill:
        try:
            # 使用SIGKILL信号强制、立即终止进程
            os.kill(pid, signal.SIGKILL)
            killed_count += 1
        except ProcessLookupError:
            logger.warning(f"【进程清理】尝试终止PID {pid} 时失败，进程已不存在。")
        except Exception as e:
            logger.error(f"【进程清理】终止PID {pid} 时发生未知错误: {e}")

    if killed_count > 0:
        logger.info(f"【进程清理】成功终止了 {killed_count} 个DeGirum工作进程。")