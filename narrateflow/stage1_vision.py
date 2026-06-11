"""
Stage 1: 视频结构化理解（VLM 独占）
- PySceneDetect 物理切片
- 动态帧差抽帧（硬上限 12 帧，长边 600px）
- VLM 推理（Provider 可插拔：openai/ollama/mock）
输出: output/{project_id}/video_meta.ai.json + frames/
"""
import json
import logging
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
from PIL import Image

from narrateflow.utils.schema_validator import extract_json_from_text, validate_vlm_output
from narrateflow.utils.ffprobe import get_video_info
from narrateflow.utils.memory_guard import log_memory_status, check_memory_safe

# 默认 System Prompt 路径
_VLM_PROMPT_PATH = Path(__file__).parent / "prompts" / "vlm_system.txt"

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 物理切片：PySceneDetect
# ═══════════════════════════════════════════════════════════════

def detect_scenes(
    video_path: str,
    threshold: float = 27.0,
    min_scene_len: int = 15,
    max_scene_duration: float = 30.0
) -> List[Dict]:
    """
    用 PySceneDetect 检测镜头边界。
    长镜头(>max_scene_duration)做二级切分。
    
    Returns:
        [{window_id, start_time, end_time, duration, start_frame, end_frame}, ...]
    """
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector
    
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold, min_scene_len=min_scene_len)
    )
    scene_manager.detect_scenes(video)
    raw_scenes = scene_manager.get_scene_list()
    
    if not raw_scenes:
        # 没有检测到镜头变化，整个视频作为一个窗口
        fps = video.frame_rate
        duration = len(video) / fps if hasattr(video, '__len__') else 0
        raw_scenes = [(0, int(duration * fps))]
    else:
        # 转换为 time 格式
        raw_scenes = [
            (s[0].get_seconds(), s[1].get_seconds())
            for s in raw_scenes
        ]
    
    # 获取 fps
    info = get_video_info(video_path)
    fps = info["fps"]
    
    # 二级切分：长镜头(>30s) 做均匀切分
    windows = []
    window_id = 0
    
    for start_t, end_t in raw_scenes:
        duration = end_t - start_t
        if duration <= max_scene_duration:
            windows.append({
                "window_id": window_id,
                "start_time": round(start_t, 3),
                "end_time": round(end_t, 3),
                "duration": round(duration, 3),
                "start_frame": int(start_t * fps),
                "end_frame": int(end_t * fps),
            })
            window_id += 1
        else:
            # 二级切分成等长段
            num_splits = max(2, int(duration / max_scene_duration) + 1)
            split_duration = duration / num_splits
            for i in range(num_splits):
                seg_start = start_t + i * split_duration
                seg_end = min(end_t, start_t + (i + 1) * split_duration)
                windows.append({
                    "window_id": window_id,
                    "start_time": round(seg_start, 3),
                    "end_time": round(seg_end, 3),
                    "duration": round(seg_end - seg_start, 3),
                    "start_frame": int(seg_start * fps),
                    "end_frame": int(seg_end * fps),
                })
                window_id += 1
    
    logger.info(f"PySceneDetect: 检测到 {len(raw_scenes)} 个镜头 → {len(windows)} 个窗口")
    return windows


# ═══════════════════════════════════════════════════════════════
# 抽帧引擎
# ═══════════════════════════════════════════════════════════════

def _frame_difference(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """计算两帧之间的归一化差异"""
    if frame1.shape != frame2.shape:
        return 1.0
    diff = np.abs(frame1.astype(np.float32) - frame2.astype(np.float32))
    return float(np.mean(diff) / 255.0)


def _resize_frame(frame: np.ndarray, max_long_edge: int = 600) -> np.ndarray:
    """将帧长边缩放到指定像素，等比缩放"""
    from PIL import Image
    h, w = frame.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_long_edge:
        return frame
    
    scale = max_long_edge / long_edge
    new_w, new_h = int(w * scale), int(h * scale)
    pil_img = Image.fromarray(frame)
    pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
    return np.array(pil_img)


def extract_frames(
    video_path: str,
    windows: List[Dict],
    output_dir: str,
    max_frames: int = 12,
    max_long_edge: int = 600,
    short_window_frames: int = 3,
    motion_threshold: float = 0.15,
    jpeg_quality: int = 85,
) -> List[Dict]:
    """
    为每个窗口抽取关键帧。
    
    策略:
    - 短窗口(≤5s): 首中尾三帧
    - 长窗口(>5s): 均匀抽取 + 动态帧差去重
    - 硬上限: max_frames 帧
    - 长边统一缩放: max_long_edge px
    
    Returns:
        更新后的 windows 列表，每项增加 frames 字段
    """
    import cv2
    
    frames_dir = Path(output_dir) / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    for window in windows:
        w_id = window["window_id"]
        start_frame = window["start_frame"]
        end_frame = window["end_frame"]
        num_frames = max(1, end_frame - start_frame + 1)
        window_duration = window["duration"]
        
        frame_paths = []
        
        if window_duration <= 5.0:
            # 短窗口：首中尾
            indices = [start_frame]
            if num_frames >= 2:
                indices.append((start_frame + end_frame) // 2)
            if num_frames >= 3:
                indices.append(end_frame)
            # 去重
            indices = sorted(set(indices))
        else:
            # 长窗口：均匀抽取 → 动态帧差去重
            # Step 1: 均匀抽取候选帧 (2倍预算)
            budget = min(max_frames * 2, num_frames)
            if budget < 1:
                budget = 1
            uniform_indices = np.linspace(start_frame, end_frame, budget, dtype=int)
            uniform_indices = sorted(set(uniform_indices.tolist()))
            
            # Step 2: 读取候选帧
            candidates = []
            for fidx in uniform_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
                ret, frame = cap.read()
                if ret:
                    candidates.append((fidx, frame))
            
            # Step 3: 动态帧差去重
            if len(candidates) <= max_frames:
                indices = [c[0] for c in candidates]
            else:
                # 保留第一帧
                kept = [candidates[0]]
                for i in range(1, len(candidates)):
                    _, curr = candidates[i]
                    _, prev = kept[-1]
                    diff = _frame_difference(curr, prev)
                    if diff >= motion_threshold:
                        kept.append(candidates[i])
                    # 如果保留太多，做间隔采样
                    if len(kept) >= max_frames:
                        # 把剩余候选均匀采样填满
                        remaining = candidates[i+1:]
                        if remaining and len(kept) < max_frames:
                            step = max(1, len(remaining) // (max_frames - len(kept)))
                            kept.extend(remaining[::step][:max_frames - len(kept)])
                        break
                
                # 如果还不够 max_frames，从未选中的补充
                if len(kept) < max_frames:
                    kept_set = {k[0] for k in kept}
                    for c in candidates:
                        if c[0] not in kept_set:
                            kept.append(c)
                            kept_set.add(c[0])
                            if len(kept) >= max_frames:
                                break
                
                # 按帧序号排序
                kept.sort(key=lambda x: x[0])
                indices = [k[0] for k in kept[:max_frames]]
        
        # Step 4: 读取最终选中的帧并保存
        for i, fidx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # 缩放
            frame = _resize_frame(frame, max_long_edge)
            
            # 保存为 JPEG
            frame_file = f"w{w_id:03d}_f{i:02d}.jpg"
            frame_path = frames_dir / frame_file
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            frame_paths.append(str(frame_path.relative_to(output_dir)))
        
        window["frames"] = frame_paths
        logger.debug(f"  窗口 {w_id}: {len(frame_paths)} 帧 ({window_duration:.1f}s)")
    
    cap.release()
    logger.info(f"抽帧完成: {len(windows)} 窗口, 总计 {sum(len(w['frames']) for w in windows)} 帧")
    return windows


# ═══════════════════════════════════════════════════════════════
# VLM Provider 接口 (可插拔)
# ═══════════════════════════════════════════════════════════════

class VLMProvider:
    """VLM 抽象基类"""
    
    def analyze_window(self, frames: List[str], system_prompt: str, 
                       window_id: int, context: Dict = None) -> Dict:
        """
        分析一个视频窗口的帧序列，返回结构化 JSON。
        
        Returns:
            vlm_output dict with: visual_summary, physical_tags, etc.
        """
        raise NotImplementedError


class MockVLMProvider(VLMProvider):
    """Mock VLM Provider — 用于 P0 快速链路测试"""
    
    def analyze_window(self, frames, system_prompt, window_id, context=None):
        """返回模拟的结构化输出"""
        return {
            "visual_summary": f"视频窗口 {window_id}，包含界面操作演示内容",
            "physical_tags": ["界面", "操作", "演示"],
            "mood_tags": ["专业", "清晰"],
            "visible_text": [],
            "action_candidates": [
                {"description": "内容展示", "time_hint": 0.0, "confidence": 0.9}
            ],
            "uncertainty": "mock provider, 非真实 VLM 输出"
        }


class OpenAIProvider(VLMProvider):
    """OpenAI 云端 VLM Provider — P0 快速验证"""
    
    def __init__(self, api_key: str = None, base_url: str = None, 
                 model: str = "gpt-4o", **kwargs):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.kwargs = kwargs
    
    def analyze_window(self, frames, system_prompt, window_id, context=None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai 未安装: pip install openai")
        
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        
        # 构建消息：文字 prompt + 图片
        content = [{"type": "text", "text": system_prompt}]
        
        for frame_path in frames:
            import base64
            # frame_path 是相对路径，需要找到绝对路径
            # 这里假设调用方会传入绝对路径或可访问的相对路径
            with open(frame_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
            })
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=self.kwargs.get("num_predict", 512),
            temperature=self.kwargs.get("temperature", 0.3),
        )
        
        raw_text = response.choices[0].message.content
        vlm_output = extract_json_from_text(raw_text)
        
        if vlm_output is None:
            return {
                "visual_summary": f"JSON 解析失败 (窗口 {window_id})",
                "physical_tags": [],
                "mood_tags": [],
                "visible_text": [],
                "action_candidates": [],
                "uncertainty": f"原始输出: {raw_text[:200]}",
                "_parse_error": "无法提取 JSON",
            }
        
        return vlm_output


# ═══════════════════════════════════════════════════════════════
# System Prompt 模板
# ═══════════════════════════════════════════════════════════════

def _load_system_prompt(prompt_path: Optional[str] = None) -> str:
    """加载 VLM System Prompt"""
    path = Path(prompt_path) if prompt_path else _VLM_PROMPT_PATH
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    # 兜底内置 prompt
    logger.warning(f"System Prompt 文件不存在: {path}，使用内置版本")
    return """你是一个视频分析助手。你将看到来自同一段视频的几帧连续截图。
请输出 JSON 格式的画面描述，包含 visual_summary, physical_tags, mood_tags,
visible_text, action_candidates, uncertainty 字段。"""


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def run_stage1(
    video_path: str = "input/video.mp4",
    manifest_path: str = "output/default/project_manifest.json",
    output_dir: str = "output/default",
    config: Optional[Dict] = None,
    vlm_provider: Optional[VLMProvider] = None,
    system_prompt_path: Optional[str] = None,
) -> str:
    """
    Stage 1 主入口。

    Args:
        video_path: 输入视频路径
        manifest_path: Stage 0 输出的 manifest
        output_dir: 输出目录
        config: 全局配置
        vlm_provider: VLM 实现（默认 Mock）
        system_prompt_path: 自定义 System Prompt 文件路径

    Returns:
        video_meta.ai.json 的路径
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stage1_cfg = config.get("stage1", {}) if config else {}
    scene_cfg = stage1_cfg.get("scenedetect", {})
    frame_cfg = stage1_cfg.get("frame_extraction", {})
    vlm_cfg = stage1_cfg.get("vlm", {})
    parse_cfg = stage1_cfg.get("parse", {})
    
    logger.info("=" * 50)
    logger.info("Stage 1: 视频结构化理解")
    logger.info("=" * 50)
    
    # 1. 读取 manifest (获取视频信息)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    project_id = manifest["project_id"]
    video_info = manifest["video"]
    
    # 2. PySceneDetect 物理切片
    windows = detect_scenes(
        str(video_path),
        threshold=scene_cfg.get("threshold", 27.0),
        min_scene_len=scene_cfg.get("min_scene_len", 15),
        max_scene_duration=scene_cfg.get("max_scene_duration", 30.0),
    )
    
    # 3. 抽帧
    windows = extract_frames(
        str(video_path),
        windows,
        str(output_dir),
        max_frames=frame_cfg.get("max_frames_per_window", 12),
        max_long_edge=frame_cfg.get("max_long_edge_px", 600),
        short_window_frames=frame_cfg.get("short_window_frames", 3),
        motion_threshold=frame_cfg.get("motion_threshold", 0.15),
        jpeg_quality=frame_cfg.get("jpeg_quality", 85),
    )
    
    # 4. VLM 推理
    if vlm_provider is None:
        # 尝试自动选择 Provider
        vlm_provider_type = stage1_cfg.get("vlm_provider", "mock")
        if vlm_provider_type == "ollama":
            try:
                from narrateflow.providers.vlm_ollama import OllamaVLMProvider
                vlm_provider = OllamaVLMProvider(
                    model=vlm_cfg.get("model_name_qwen", "qwen3-vl:8b"),
                    num_ctx=vlm_cfg.get("num_ctx", 4096),
                    num_predict=vlm_cfg.get("num_predict", 512),
                    temperature=vlm_cfg.get("temperature", 0.3),
                    keep_alive=vlm_cfg.get("keep_alive", 0),
                )
                logger.info("✓ Ollama VLM Provider 已初始化")
            except Exception as e:
                logger.warning(f"Ollama 初始化失败: {e}，降级为 Mock")
                vlm_provider = MockVLMProvider()
        else:
            logger.warning("未提供 VLM Provider，使用 Mock 模式（非真实 VLM 输出）")
            vlm_provider = MockVLMProvider()

    provider_name = vlm_provider.name if hasattr(vlm_provider, 'name') else vlm_provider.__class__.__name__
    logger.info(f"VLM Provider: {provider_name}")

    # 加载 System Prompt
    system_prompt = _load_system_prompt(system_prompt_path)
    logger.debug(f"System Prompt: {len(system_prompt)} 字符")

    # 内存检查（启动前记录基线）
    log_memory_status("Stage1 启动")

    total_windows = len(windows)
    parsed_count = 0
    error_count = 0
    t_start = time.time()

    for i, window in enumerate(windows):
        w_id = window["window_id"]
        frame_rel_paths = window["frames"]
        # 转换为绝对路径
        frame_abs_paths = [str(output_dir / p) for p in frame_rel_paths]

        logger.info(f"  推理窗口 {w_id+1}/{total_windows} "
                    f"({window['duration']:.1f}s, {len(frame_abs_paths)} 帧)...")

        for attempt in range(parse_cfg.get("max_retries_per_window", 2) + 1):
            try:
                vlm_output = vlm_provider.analyze_window(
                    frame_abs_paths, system_prompt, w_id
                )
                
                # Schema 校验
                is_valid, error = validate_vlm_output(vlm_output)
                if is_valid:
                    window["vlm_output"] = vlm_output
                    window["needs_review"] = False
                    parsed_count += 1
                    break
                else:
                    if attempt == parse_cfg.get("max_retries_per_window", 2):
                        window["vlm_output"] = vlm_output
                        window["needs_review"] = True
                        window["parse_error"] = error
                        error_count += 1
                    else:
                        logger.warning(f"    校验失败 (尝试 {attempt+1}): {error}")
            except Exception as e:
                logger.error(f"    推理失败 (尝试 {attempt+1}): {e}")
                if attempt == parse_cfg.get("max_retries_per_window", 2):
                    window["vlm_output"] = {
                        "visual_summary": f"VLM 推理失败: {str(e)[:100]}",
                        "physical_tags": [],
                        "mood_tags": [],
                        "visible_text": [],
                        "action_candidates": [],
                        "uncertainty": f"错误: {e}",
                    }
                    window["needs_review"] = True
                    window["parse_error"] = str(e)
                    error_count += 1
    
    t_elapsed = time.time() - t_start
    log_memory_status("Stage1 推理完成")
    logger.info(f"VLM 推理完成: {parsed_count} 成功, {error_count} 失败, "
                f"耗时 {t_elapsed:.1f}s")

    # 释放 VLM（Mac 可常驻但保留此安全机制）
    if hasattr(vlm_provider, 'release'):
        vlm_provider.release()
        log_memory_status("Stage1 模型释放后")
    
    # 5. 组装输出
    video_meta = {
        "project_id": project_id,
        "model": provider_name,
        "generated_at": datetime.now().isoformat(),
        "stats": {
            "total_windows": total_windows,
            "total_frames": sum(len(w["frames"]) for w in windows),
            "parsed_ok": parsed_count,
            "parse_errors": error_count,
            "elapsed_seconds": round(t_elapsed, 1),
            "video_duration": video_info["duration"],
        },
        "windows": windows,
    }
    
    # 6. 输出
    output_path = output_dir / "video_meta.ai.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(video_meta, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✓ video_meta.ai.json 已输出: {output_path}")
    logger.info(f"  窗口数: {total_windows}, 帧数: {video_meta['stats']['total_frames']}")
    logger.info(f"  解析成功率: {parsed_count}/{total_windows}")
    
    return str(output_path)


# ═══════════════════════════════════════════════════════════════
# CLI 测试入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    video = sys.argv[1] if len(sys.argv) > 1 else "input/video.mp4"
    manifest = sys.argv[2] if len(sys.argv) > 2 else "output/default/project_manifest.json"
    output = sys.argv[3] if len(sys.argv) > 3 else "output/default"
    
    try:
        vmp = run_stage1(video, manifest, output)
        print(f"\n✓ Stage 1 完成: {vmp}")
    except Exception as e:
        logger.error(f"Stage 1 失败: {e}", exc_info=True)
        sys.exit(1)
