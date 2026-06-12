"""
VoxCPM2 TTS Provider — Mac M4 Pro / RTX 3060 双平台
统一接口，自动检测平台选择 dtype。
"""
import logging
import subprocess
import time
from pathlib import Path
from typing import Optional, List

from narrateflow.providers.tts_base import TTSProvider

logger = logging.getLogger(__name__)


class VoxCPM2Provider(TTSProvider):
    """
    VoxCPM2 TTS Provider (2B参数, 48kHz)。

    平台自动适配:
    - Mac MPS: 使用 FP16（禁止 BF16，否则报错/降级 CPU）
    - Windows CUDA (RTX 3060): 使用 BF16（Ampere 架构原生加速，约 4GB 显存）
    - CPU 兜底: FP32

    安装: pip install git+https://github.com/OpenBMB/VoxCPM.git
    首次运行会自动下载模型 (~2.5GB)。
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        dtype: Optional[str] = None,
        device: Optional[str] = None,
        sample_rate: int = 48000,
    ):
        """
        Args:
            model_path: 模型路径 (None = 自动下载)
            dtype: 强制指定 dtype (None = 自动检测: Mac→fp16, CUDA→bf16)
            device: 强制指定 device (None = 自动检测)
            sample_rate: 输出采样率 (VoxCPM2 原生 48kHz)
        """
        self.model_path = model_path
        self._sample_rate = sample_rate
        self._model = None
        self._available = False
        self._dtype = dtype
        self._device = device

        # 自动检测平台
        import torch
        self._is_cuda = torch.cuda.is_available()
        self._is_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

        # 自动选择 dtype
        if self._dtype is None:
            if self._is_cuda:
                # RTX 3060: BF16 (Ampere 原生加速)
                self._dtype = "bf16"
                logger.info("CUDA 检测到 → 使用 BF16")
            elif self._is_mps:
                # Mac MPS: 必须 FP16（MPS 不支持 BF16）
                self._dtype = "fp16"
                logger.info("MPS 检测到 → 使用 FP16 (Mac MPS 禁止 BF16)")
            else:
                self._dtype = "fp32"
                logger.info("CPU 模式 → 使用 FP32")

        # 尝试加载
        self._try_load()

    @property
    def name(self) -> str:
        return "voxcpm2"

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def _try_load(self):
        """尝试加载 VoxCPM2 模型"""
        try:
            from voxcpm import VoxCPM
            self._model = VoxCPM.from_pretrained(
                self.model_path or "OpenBMB/VoxCPM2-0.5B",
                dtype=self._dtype,
                device=self._device,
            )
            self._available = True
            logger.info(f"VoxCPM2 加载成功 (dtype={self._dtype})")
        except ImportError:
            logger.warning(
                "VoxCPM2 未安装。安装命令:\n"
                "  pip install git+https://github.com/OpenBMB/VoxCPM.git\n"
                "当前将降级使用 edge-tts。"
            )
        except Exception as e:
            logger.warning(f"VoxCPM2 加载失败: {e}")

    def check_available(self) -> bool:
        return self._available

    def synthesize(self, text: str, output_path: str, **kwargs) -> float:
        """
        使用 VoxCPM2 合成音频。

        Mac: 自动使用 FP16 + MPS 后端
        3060: 自动使用 BF16 + CUDA 后端
        输出: 48kHz WAV
        """
        if not self._available:
            raise RuntimeError(
                "VoxCPM2 未加载。请先安装:\n"
                "  pip install git+https://github.com/OpenBMB/VoxCPM.git"
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        t_start = time.time()

        # VoxCPM2 推理
        # 根据 repo 的实际 API 调整（以下是预期接口）
        voice = kwargs.get("voice", "default")
        speed = kwargs.get("speed", 1.0)

        try:
            # VoxCPM2 API (预期):
            # audio = model.generate(text, voice=voice, speed=speed)
            # sf.write(output_path, audio, self.sample_rate)
            audio = self._model.generate(
                text=text,
                voice=voice,
                speed=speed,
            )

            import soundfile as sf
            sf.write(str(output_path), audio, self.sample_rate)
            duration = len(audio) / self.sample_rate

        except Exception as e:
            raise RuntimeError(f"VoxCPM2 合成失败: {e}")

        t_elapsed = time.time() - t_start
        logger.debug(f"VoxCPM2: {duration:.1f}s audio, {t_elapsed:.1f}s "
                     f"(ratio={t_elapsed/duration:.1f}x)")

        return round(duration, 3)

    def batch_synthesize(
        self,
        texts: List[str],
        output_dir: str,
        prefix: str = "seg",
        **kwargs
    ) -> List[float]:
        """批量合成，保持音色一致性（同一次模型加载）"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        durations = []
        for i, text in enumerate(texts):
            out_path = output_dir / f"{prefix}_{i:03d}.wav"
            dur = self.synthesize(text, str(out_path), **kwargs)
            durations.append(dur)
            logger.info(f"  [{i+1}/{len(texts)}] {dur:.1f}s")

        return durations

    def release(self):
        """释放 GPU 显存"""
        if self._model is not None:
            del self._model
            self._model = None
            self._available = False
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("VoxCPM2 模型已释放")
