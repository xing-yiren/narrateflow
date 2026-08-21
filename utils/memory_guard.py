"""
内存监控与自动降级 — Mac M4 Pro
Mac 使用统一内存，通过 psutil 监控物理内存使用率。
"""
import logging
import time
import psutil
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def get_memory_usage() -> Tuple[float, float]:
    """
    获取当前内存使用情况。

    Mac 统一内存: 使用 (total - available) 作为真实占用，
    而非 psutil.used（不包含 wired/compressed）。

    Returns:
        (used_gb, total_gb)
    """
    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024 ** 3)
    # Mac: total - available = wired + active + compressed + resident
    used_gb = (mem.total - mem.available) / (1024 ** 3)
    return round(used_gb, 2), round(total_gb, 2)


def get_memory_percent() -> float:
    """获取真实内存占用百分比 = (total - available) / total × 100"""
    mem = psutil.virtual_memory()
    return round((mem.total - mem.available) / mem.total * 100, 1)


def wait_for_memory_release(
    target_percent: float = 50.0,
    timeout: float = 30.0,
    check_interval: float = 1.0
) -> bool:
    """
    轮询等待内存使用率降到目标值以下。
    
    Mac 统一内存由系统托管，进程死掉后可能有 Page Cache 残留，
    必须用 psutil 确认物理内存真正释放（而非仅看进程退出）。
    
    Args:
        target_percent: 目标内存使用率百分比
        timeout: 最长等待时间(秒)
        check_interval: 检查间隔(秒)
        
    Returns:
        True 如果内存降到安全水位，False 如果超时
    """
    t_start = time.time()
    
    while time.time() - t_start < timeout:
        current = get_memory_percent()
        used, total = get_memory_usage()
        
        if current <= target_percent:
            logger.info(f"内存安全: {current:.1f}% ({used:.1f}/{total:.1f} GB)")
            return True
        
        logger.debug(f"等待内存释放... {current:.1f}% ({used:.1f}/{total:.1f} GB)")
        time.sleep(check_interval)
    
    used, total = get_memory_usage()
    logger.warning(f"内存释放超时 ({timeout}s): {get_memory_percent():.1f}% ({used:.1f}/{total:.1f} GB)")
    return False


def check_memory_safe(threshold_percent: float = 80.0) -> bool:
    """
    检查当前内存是否在安全水位以下。
    
    Returns:
        True 如果内存安全
    """
    current = get_memory_percent()
    if current > threshold_percent:
        used, total = get_memory_usage()
        logger.warning(f"内存超阈值: {current:.1f}% > {threshold_percent:.1f}% "
                       f"({used:.1f}/{total:.1f} GB)")
        return False
    return True


def log_memory_status(tag: str = ""):
    """记录当前内存状态"""
    used, total = get_memory_usage()
    percent = get_memory_percent()
    logger.info(f"[内存] {tag}: {percent:.1f}% ({used:.1f}/{total:.1f} GB)")


class MemoryContext:
    """内存上下文管理器 — 进入/退出时记录内存"""
    
    def __init__(self, label: str = ""):
        self.label = label
        self.start_used = 0.0
        self.start_total = 0.0
    
    def __enter__(self):
        self.start_used, self.start_total = get_memory_usage()
        logger.info(f"[内存] 进入 {self.label}: {get_memory_percent():.1f}% "
                    f"({self.start_used:.1f}/{self.start_total:.1f} GB)")
        return self
    
    def __exit__(self, *args):
        end_used, _ = get_memory_usage()
        delta = end_used - self.start_used
        direction = "↑" if delta > 0 else "↓"
        logger.info(f"[内存] 退出 {self.label}: {get_memory_percent():.1f}% "
                    f"({end_used:.1f}/{self.start_total:.1f} GB, {direction}{abs(delta):.2f} GB)")
