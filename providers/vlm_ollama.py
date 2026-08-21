"""
Ollama VLM Provider — 本地 Qwen3-VL 部署
通过 Ollama API 与本地模型交互。
"""
import json
import logging
import time
import base64
from pathlib import Path
from typing import Dict, List, Optional

import requests

from narrateflow.providers.vlm_base import VLMProvider

logger = logging.getLogger(__name__)


class OllamaVLMProvider(VLMProvider):
    """Ollama 本地 VLM Provider — Qwen3-VL-8B"""

    # Qwen3-VL Thinking 模式内部推理预算（不可见 tokens）
    # num_predict 必须 > THINKING_OVERHEAD + 预期输出 tokens
    THINKING_OVERHEAD = 512
    MIN_PREDICT_VISION = 1536  # Thinking 内部推理 ~900tok，≥1536 保证完整 JSON

    def __init__(
        self,
        model: str = "qwen3-vl:8b",
        base_url: str = "http://localhost:11434",
        num_ctx: int = 4096,
        num_predict: int = 1024,  # Thinking 模型需要更高预算
        temperature: float = 0.3,
        keep_alive: int = 0,
        timeout: int = 180,
        auto_adjust_predict: bool = True,  # 空响应时自动提高 num_predict
    ):
        """
        Args:
            model: Ollama 模型名
            base_url: Ollama API 地址
            num_ctx: 上下文窗口大小 (Mac可尝试放宽)
            num_predict: 最大生成 token 数 (Thinking模型需≥1024)
            temperature: 采样温度
            keep_alive: 模型驻留时间(秒)，0=立即释放，-1=常驻
            timeout: 单次推理超时(秒)
            auto_adjust_predict: 空响应时自动提高 num_predict 重试
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.num_ctx = num_ctx
        self.num_predict = max(num_predict, self.MIN_PREDICT_VISION)
        self.temperature = temperature
        self.keep_alive = keep_alive
        self.timeout = timeout
        self.auto_adjust_predict = auto_adjust_predict

        # 检测是否为 thinking 模型
        self.is_thinking_model = "thinking" in model.lower() or "qwen3" in model.lower()
        if self.is_thinking_model and num_predict < self.MIN_PREDICT_VISION:
            logger.warning(
                f"Thinking 模型 '{model}' 建议 num_predict ≥ {self.MIN_PREDICT_VISION}，"
                f"当前: {num_predict}（可能产生空输出）"
            )

        # 检查 Ollama 是否可用
        self._check_connection()
    
    @property
    def name(self) -> str:
        return f"ollama:{self.model}"
    
    def _check_connection(self):
        """检查 Ollama 服务是否可达"""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            
            if self.model not in models:
                logger.warning(
                    f"模型 '{self.model}' 未找到，可用模型: {models}。"
                    f"请运行: ollama pull {self.model}"
                )
            else:
                logger.info(f"Ollama 连接成功，模型: {self.model}")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"无法连接 Ollama ({self.base_url})，请确保服务已启动:\n"
                f"  brew services start ollama"
            )
        except Exception as e:
            raise RuntimeError(f"Ollama 连接检查失败: {e}")
    
    def analyze_window(
        self,
        frames: List[str],
        system_prompt: str,
        window_id: int,
        context: Optional[Dict] = None,
    ) -> Dict:
        """
        向 Ollama 发送帧序列 + 提示词，获取结构化 JSON。
        
        Qwen3-VL 通过 Ollama 的 images 参数接收图片（base64）。
        """
        # 读取帧图片为 base64
        images_base64 = []
        for frame_path in frames:
            p = Path(frame_path)
            if not p.exists():
                logger.warning(f"帧文件不存在: {frame_path}")
                continue
            with open(p, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
            images_base64.append(img_b64)
        
        if not images_base64:
            return {
                "visual_summary": f"无可用帧 (窗口 {window_id})",
                "physical_tags": [],
                "mood_tags": [],
                "visible_text": [],
                "action_candidates": [],
                "uncertainty": "所有帧文件缺失",
            }
        
        # 构建 Ollama API 请求（支持 Thinking 模型空响应自动重试）
        predict_budgets = [self.num_predict]
        if self.auto_adjust_predict and self.is_thinking_model:
            # Thinking 模型需要预留内部推理预算
            if self.num_predict < 1536:
                predict_budgets = [self.num_predict, 1536]
            else:
                predict_budgets = [self.num_predict, self.num_predict * 2]

        raw_text = ""
        eval_count = 0
        eval_duration = 0
        t_elapsed = 0
        last_error = None

        for attempt, np_val in enumerate(predict_budgets):
            payload = {
                "model": self.model,
                "prompt": system_prompt,
                "images": images_base64,
                "stream": False,
                "options": {
                    "num_ctx": self.num_ctx,
                    "num_predict": np_val,
                    "temperature": self.temperature,
                },
                "keep_alive": self.keep_alive,
            }

            t_start = time.time()
            try:
                resp = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
            except requests.exceptions.Timeout:
                last_error = f"推理超时 ({self.timeout}s)"
                logger.error(f"Ollama {last_error} - 窗口 {window_id}")
                continue
            except requests.exceptions.ConnectionError:
                last_error = "Ollama 连接断开"
                logger.error(f"{last_error} - 窗口 {window_id}")
                continue
            except Exception as e:
                last_error = str(e)
                logger.error(f"Ollama 请求失败 - 窗口 {window_id}: {e}")
                continue

            t_elapsed = time.time() - t_start
            data = resp.json()
            raw_text = data.get("response", "").strip()
            eval_count = data.get("eval_count", 0)
            eval_duration = data.get("eval_duration", 0)

            # 检测 Thinking 模式空输出：eval 了大量 token 但无可见输出
            if not raw_text and eval_count > 50:
                logger.warning(
                    f"  窗口 {window_id}: Thinking 模式耗尽 {eval_count} tokens 但无可见输出"
                    f" (num_predict={np_val})"
                )
                if self.auto_adjust_predict and attempt < len(predict_budgets) - 1:
                    next_np = predict_budgets[attempt + 1]
                    logger.info(f"  → 自动提高 num_predict: {np_val} → {next_np}，重试...")
                    continue

            break  # Got response or exhausted retries

        if last_error and not raw_text:
            return self._error_output(window_id, last_error)

        # 记录性能
        eval_dur_s = eval_duration / 1e9 if eval_duration else 0
        tokens_per_sec = eval_count / eval_dur_s if eval_dur_s > 0 else 0
        visible_chars = len(raw_text)

        logger.debug(
            f"  窗口 {window_id}: {t_elapsed:.1f}s, "
            f"{eval_count} tokens ({visible_chars} visible) @ {tokens_per_sec:.1f} tok/s"
        )

        # 如果最终仍然是空响应
        if not raw_text:
            return {
                "visual_summary": f"模型返回空输出 (窗口 {window_id}, eval={eval_count} tokens)",
                "physical_tags": [],
                "mood_tags": [],
                "visible_text": [],
                "action_candidates": [],
                "uncertainty": f"Thinking 模式消耗了全部 {eval_count} tokens 预算",
                "_parse_error": "空输出",
            }

        # 解析 JSON（多级容错）+ 字段名归一化
        from narrateflow.utils.schema_validator import extract_json_from_text

        vlm_output = extract_json_from_text(raw_text)

        if vlm_output is None:
            return {
                "visual_summary": f"JSON 解析失败 (窗口 {window_id})",
                "physical_tags": [],
                "mood_tags": [],
                "visible_text": [],
                "action_candidates": [],
                "uncertainty": f"原始输出前200字符: {raw_text[:200]}",
                "_parse_error": "无法提取 JSON",
                "_raw_output": raw_text[:500],
            }

        return _normalize_vlm_output(vlm_output)

    def release(self):
        """释放模型显存"""
        try:
            requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": "",
                    "keep_alive": 0,
                },
                timeout=5,
            )
            logger.info(f"Ollama 模型 {self.model} 已释放")
        except Exception as e:
            logger.warning(f"释放模型时出错: {e}")
    
    def _error_output(self, window_id: int, error_msg: str) -> Dict:
        return {
            "visual_summary": f"VLM 推理失败 (窗口 {window_id})",
            "physical_tags": [],
            "mood_tags": [],
            "visible_text": [],
            "action_candidates": [],
            "uncertainty": f"错误: {error_msg}",
            "_parse_error": error_msg,
        }


def check_ollama_available(base_url: str = "http://localhost:11434") -> bool:
    """快速检查 Ollama 是否可用"""
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def get_model_size(model_name: str, base_url: str = "http://localhost:11434") -> Optional[int]:
    """获取模型大小（字节）"""
    try:
        resp = requests.post(
            f"{base_url}/api/show",
            json={"name": model_name},
            timeout=10,
        )
        data = resp.json()
        return data.get("size", 0)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════
# 字段名归一化（模型输出 → 标准 Schema）
# ═══════════════════════════════════════════════════════════════

_FIELD_MAP = {
    "summary": "visual_summary",
    "visual_summary": "visual_summary",
    "description": "visual_summary",
    "tags": "physical_tags",
    "physical_tags": "physical_tags",
    "objects": "physical_tags",
    "mood": "mood_tags",
    "mood_tags": "mood_tags",
    "atmosphere": "mood_tags",
    "text": "visible_text",
    "visible_text": "visible_text",
    "ocr": "visible_text",
    "actions": "action_candidates",
    "action_candidates": "action_candidates",
    "uncertainty": "uncertainty",
}

_ACTION_FIELD_MAP = {
    "desc": "description",
    "description": "description",
    "action": "description",
    "confidence": "confidence",
    "conf": "confidence",
    "time_hint": "time_hint",
}


def _normalize_vlm_output(raw: dict) -> dict:
    """将模型输出的任意字段名映射为标准 Schema 字段名"""
    output = {}

    for key, value in raw.items():
        mapped = _FIELD_MAP.get(key, key)
        if mapped == "action_candidates" and isinstance(value, list):
            normalized_actions = []
            for action in value:
                if isinstance(action, dict):
                    norm_action = {}
                    for ak, av in action.items():
                        norm_action[_ACTION_FIELD_MAP.get(ak, ak)] = av
                    if "confidence" in norm_action:
                        try:
                            norm_action["confidence"] = float(norm_action["confidence"])
                        except (ValueError, TypeError):
                            norm_action["confidence"] = 0.5
                    if "time_hint" not in norm_action:
                        norm_action["time_hint"] = 0.0
                    normalized_actions.append(norm_action)
            output[mapped] = normalized_actions
        elif mapped == "mood_tags" and isinstance(value, str):
            output[mapped] = [t.strip() for t in value.replace("、", ",").split(",") if t.strip()]
        else:
            output[mapped] = value

    # 确保必填字段存在
    for field in ["visual_summary", "physical_tags", "mood_tags",
                   "visible_text", "action_candidates", "uncertainty"]:
        if field not in output:
            if field in ("physical_tags", "mood_tags", "visible_text",
                         "action_candidates"):
                output[field] = []
            elif field == "uncertainty":
                output[field] = "无"
            else:
                output[field] = ""

    return output
