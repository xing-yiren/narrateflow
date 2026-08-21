"""
TTS 抽象接口
所有 TTS Provider 必须实现此接口。
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional


class TTSProvider(ABC):
    """TTS 抽象基类"""

    @abstractmethod
    def synthesize(self, text: str, output_path: str, **kwargs) -> float:
        """
        合成一段音频。

        Args:
            text: 待合成的文本
            output_path: 输出 WAV 文件路径
            **kwargs: provider-specific 参数

        Returns:
            真实音频时长（秒），必须来自实测，禁止估算
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider 标识名"""
        ...

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """输出采样率 (Hz)"""
        ...

    def check_available(self) -> bool:
        """检查 Provider 是否可用（子类按需实现）"""
        return True

    def release(self):
        """释放模型资源（子类按需实现）"""
        pass
