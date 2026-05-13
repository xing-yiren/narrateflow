# 本地使用说明

## 1. 启动网页界面

双击 `D:\qwen3-tts\start_webui.bat`

浏览器打开 `http://127.0.0.1:7860`

## 2. 直接克隆

- 上传参考音频
- 填参考音频对应文本
- 填待合成文本
- 点击生成

输出会保存到：`D:\qwen3-tts\outputs\音色名称\时间戳__文本摘要.wav`

同目录会自动生成一份 `.json` 元数据，保存音色名、文本、语言、时间等信息。

## 3. 保存音色文件后复用

先在“保存音色文件”页上传参考音频和参考文本，保存成 `.pt` 音色文件。

以后可以在“使用音色文件生成”页直接选择这个 `.pt` 文件反复生成，不需要每次重新提取音色。

## 4. 命令行用法

检查部署状态：

```bash
venv313\Scripts\python voice_clone_tool.py check
```

直接根据参考音频生成：

```bash
venv313\Scripts\python voice_clone_tool.py clone-once ^
  --voice-name 老师女声 ^
  --ref-audio D:\path\ref.wav ^
  --ref-text "这是参考音频的文本" ^
  --text "这是要合成的新文本" ^
  --language Chinese
```

先保存音色文件：

```bash
venv313\Scripts\python voice_clone_tool.py save-prompt ^
  --voice-name 老师女声 ^
  --ref-audio D:\path\ref.wav ^
  --ref-text "这是参考音频的文本"
```

用音色文件生成：

```bash
venv313\Scripts\python voice_clone_tool.py synthesize ^
  --profile D:\qwen3-tts\outputs\voice_profiles\老师女声\老师女声.pt ^
  --text "请同学们打开课本第十页。" ^
  --language Chinese
```
