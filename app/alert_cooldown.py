"""线程安全的警报冷却管理器"""
import time
from threading import Lock


class AlertCooldown:

    # 冷却时间配置（秒）
    COOLDOWN_CONFIG = {
        # 垃圾类别（根据你的实际垃圾类别名配置，或使用默认）
        "default_garbage": 15 * 60,  # 垃圾默认 15 分钟
        "recyclable": 15 * 60,
        "kitchen_waste": 15 * 60,
        "hazardous": 15 * 60,
        "other": 15 * 60,
        # 火情/烟雾类别
        "fire": 90,  # 1.5 分钟
        "smoke": 90,
    }

    def __init__(self):
        self._last_alert_time = {}  
        self._lock = Lock()

    def _get_cooldown_seconds(self, category: str) -> int:
        """获取某类别的冷却秒数"""
        # 优先匹配精确类别，否则按垃圾默认
        if category in self.COOLDOWN_CONFIG:
            return self.COOLDOWN_CONFIG[category]
        # 如果类别名包含 fire/smoke 关键字（防御）
        if "fire" in category.lower() or "smoke" in category.lower():
            return 90
        return self.COOLDOWN_CONFIG["default_garbage"]

    def can_alert(self, category: str) -> bool:
        """
        检查该类别是否可以触发警报
        返回 True 表示允许报警（未冷却或已过冷却期）
        """
        with self._lock:
            now = time.time()
            last = self._last_alert_time.get(category, 0)
            cooldown = self._get_cooldown_seconds(category)
            if now - last >= cooldown:
                # 更新最后报警时间
                self._last_alert_time[category] = now
                return True
            else:
                remaining = int(cooldown - (now - last))
                print(f"[冷却抑制] 类别 '{category}' 还需等待 {remaining} 秒")
                return False

    def reset_category(self, category: str):
        """手动重置某个类别的冷却（用于测试或强制）"""
        with self._lock:
            self._last_alert_time.pop(category, None)


# 全局单例
cooldown_manager = AlertCooldown()