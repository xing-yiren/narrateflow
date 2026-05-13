from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import gradio as gr

from voice_clone_tool import (
    OUTPUTS_DIR,
    PROFILES_DIR,
    create_output_paths,
    load_model,
    load_prompt_file,
    save_prompt_file,
    slugify,
    write_outputs,
)


ROOT = Path(__file__).resolve().parent
_TTS = None


def get_tts():
    global _TTS
    if _TTS is None:
        _TTS = load_model()
    return _TTS


def normalize_audio_input(audio):
    if audio is None:
        return None
    if isinstance(audio, tuple) and len(audio) == 2:
        sr, wav = audio
        return wav, int(sr)
    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        return audio["data"], int(audio["sampling_rate"])
    return None


def list_profiles() -> list[str]:
    if not PROFILES_DIR.exists():
        return []
    return sorted(str(path) for path in PROFILES_DIR.glob("*/*.pt"))


def save_profile(voice_name: str, ref_audio, ref_text: str, xvector_only: bool):
    if not voice_name.strip():
        return gr.update(), "请填写音色名称。"
    audio_tuple = normalize_audio_input(ref_audio)
    if audio_tuple is None:
        return gr.update(), "请上传参考音频。"
    if (not xvector_only) and not ref_text.strip():
        return gr.update(), "未开启 x-vector only 时，必须填写参考音频文本。"

    tts = get_tts()
    prompt_items = tts.create_voice_clone_prompt(
        ref_audio=audio_tuple,
        ref_text=ref_text.strip() or None,
        x_vector_only_mode=bool(xvector_only),
    )
    profile_name = slugify(voice_name)
    profile_path = PROFILES_DIR / profile_name / f"{profile_name}.pt"
    save_prompt_file(prompt_items, profile_path)
    return gr.update(
        choices=list_profiles(), value=str(profile_path)
    ), f"已保存音色文件: {profile_path}"


def clone_direct(
    voice_name: str,
    ref_audio,
    ref_text: str,
    xvector_only: bool,
    language: str,
    text: str,
):
    if not voice_name.strip():
        return None, "请填写音色名称。"
    if not text.strip():
        return None, "请填写待合成文本。"
    audio_tuple = normalize_audio_input(ref_audio)
    if audio_tuple is None:
        return None, "请上传参考音频。"
    if (not xvector_only) and not ref_text.strip():
        return None, "未开启 x-vector only 时，必须填写参考音频文本。"

    tts = get_tts()
    wavs, sample_rate = tts.generate_voice_clone(
        text=text.strip(),
        language=language,
        ref_audio=audio_tuple,
        ref_text=ref_text.strip() or None,
        x_vector_only_mode=bool(xvector_only),
        max_new_tokens=1024,
    )
    wav_path, meta_path = create_output_paths(voice_name, text)
    write_outputs(
        wav_path,
        meta_path,
        wavs[0],
        sample_rate,
        {
            "voice_name": voice_name,
            "language": language,
            "text": text.strip(),
            "ref_text": ref_text.strip(),
            "xvector_only": bool(xvector_only),
            "wav_path": str(wav_path),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    return (sample_rate, wavs[0]), f"生成完成: {wav_path}"


def synthesize_from_profile(
    profile_path: str, language: str, text: str, voice_name: str
):
    if not profile_path:
        return None, "请选择音色文件。"
    if not text.strip():
        return None, "请填写待合成文本。"

    tts = get_tts()
    prompt_items = load_prompt_file(Path(profile_path))
    wavs, sample_rate = tts.generate_voice_clone(
        text=text.strip(),
        language=language,
        voice_clone_prompt=prompt_items,
        max_new_tokens=1024,
    )
    final_voice_name = voice_name.strip() or Path(profile_path).stem
    wav_path, meta_path = create_output_paths(final_voice_name, text)
    write_outputs(
        wav_path,
        meta_path,
        wavs[0],
        sample_rate,
        {
            "voice_name": final_voice_name,
            "language": language,
            "text": text.strip(),
            "profile_path": profile_path,
            "wav_path": str(wav_path),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    return (sample_rate, wavs[0]), f"生成完成: {wav_path}"


with gr.Blocks(title="Qwen3 Voice Clone", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Qwen3-TTS 本地声音克隆
        模型目录固定为 `D:/qwen3-tts/models`，输出目录固定为 `D:/qwen3-tts/outputs`。
        """
    )

    with gr.Tab("直接克隆"):
        voice_name_1 = gr.Textbox(label="音色名称", placeholder="例如：老师女声")
        ref_audio_1 = gr.Audio(label="参考音频", type="numpy")
        ref_text_1 = gr.Textbox(label="参考音频文本", lines=3)
        xvector_only_1 = gr.Checkbox(
            label="仅使用说话人向量（无需参考文本，但质量可能下降）", value=False
        )
        language_1 = gr.Dropdown(
            label="语言",
            choices=["Auto", "Chinese", "English", "Japanese", "Korean"],
            value="Auto",
        )
        text_1 = gr.Textbox(label="待合成文本", lines=4)
        run_1 = gr.Button("开始生成", variant="primary")
        audio_out_1 = gr.Audio(label="输出音频", type="numpy")
        status_1 = gr.Textbox(label="状态")
        run_1.click(
            clone_direct,
            inputs=[
                voice_name_1,
                ref_audio_1,
                ref_text_1,
                xvector_only_1,
                language_1,
                text_1,
            ],
            outputs=[audio_out_1, status_1],
        )

    with gr.Tab("保存音色文件"):
        voice_name_2 = gr.Textbox(label="音色名称")
        ref_audio_2 = gr.Audio(label="参考音频", type="numpy")
        ref_text_2 = gr.Textbox(label="参考音频文本", lines=3)
        xvector_only_2 = gr.Checkbox(label="仅使用说话人向量", value=False)
        save_btn = gr.Button("保存音色文件", variant="primary")
        profiles_dropdown = gr.Dropdown(
            label="已保存音色文件", choices=list_profiles(), allow_custom_value=True
        )
        status_2 = gr.Textbox(label="状态")
        save_btn.click(
            save_profile,
            inputs=[voice_name_2, ref_audio_2, ref_text_2, xvector_only_2],
            outputs=[profiles_dropdown, status_2],
        )

    with gr.Tab("使用音色文件生成"):
        profiles_dropdown_2 = gr.Dropdown(
            label="音色文件", choices=list_profiles(), allow_custom_value=True
        )
        voice_name_3 = gr.Textbox(
            label="输出音色名称", placeholder="留空则使用音色文件名"
        )
        language_3 = gr.Dropdown(
            label="语言",
            choices=["Auto", "Chinese", "English", "Japanese", "Korean"],
            value="Auto",
        )
        text_3 = gr.Textbox(label="待合成文本", lines=4)
        run_3 = gr.Button("开始生成", variant="primary")
        audio_out_3 = gr.Audio(label="输出音频", type="numpy")
        status_3 = gr.Textbox(label="状态")
        run_3.click(
            synthesize_from_profile,
            inputs=[profiles_dropdown_2, language_3, text_3, voice_name_3],
            outputs=[audio_out_3, status_3],
        )

    gr.Markdown(f"当前输出根目录: `{OUTPUTS_DIR}`")


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
