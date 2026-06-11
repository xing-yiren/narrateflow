"""
VLM 抽象接口
所有 VLM Provider 必须实现此接口。
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class VLMProvider(ABC):
    """VLM 抽象基类"""
    
    @abstractmethod
    def analyze_window(
        self,
        frames: List[str],
        system_prompt: str,
        window_id: int,
        context: Optional[Dict] = None,
    ) -> Dict:
        """
        分析一个视频窗口的帧序列，返回结构化 JSON。
        
        Args:
            frames: 帧图片文件的绝对路径列表
            system_prompt: 系统提示词
            window_id: 窗口编号
            context: 可选的上下文信息
            
        Returns:
            dict with keys: visual_summary, physical_tags, mood_tags,
                           visible_text, action_candidates, uncertainty
        """
        ...
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider 名称"""
        ...
    
    def release(self):
        """释放模型资源（子类按需实现）"""
        pass
