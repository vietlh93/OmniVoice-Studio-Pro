"""
OmniVoice Studio Pro - Gradio Web Interface
"""
import os
import warnings

# Tắt cảnh báo không cần thiết từ thư viện (PyTorch, Transformers, Gradio, TF)
warnings.filterwarnings('ignore')
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
import gradio as gr

import tempfile
from pathlib import Path
from typing import cast
from gradio.components.textbox import InputHTMLAttributes

os.environ["GRADIO_TEMP_DIR"] = tempfile.gettempdir() + "/my_gradio_tmp"
os.makedirs(os.environ["GRADIO_TEMP_DIR"], exist_ok=True)

# Clear temp folder on startup (không crash nếu lỗi)
try:
    import shutil
    temp_dir = os.environ["GRADIO_TEMP_DIR"]
    if os.path.exists(temp_dir):
        for item in os.listdir(temp_dir):
            item_path = os.path.join(temp_dir, item)
            try:
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception:
                pass
except Exception:
    pass

from OmniVoice.omnivoice_inference.ttsOmni import generate_speech_omni
from ui_app_Support.app_support.app_model_management import (
    configure_device,
    get_omni_model,
)
from ui_app_Support.app_support.app_support import (
    CSS, APP_INIT_JS,
    list_voices, get_default_voice, get_wavs_dir,
    save_path, load_path,
    save_generated_audio_and_srt,
)

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"Device: {DEVICE}")

configure_device(DEVICE)

def browse_folder(current_val):
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes('-topmost', 1)
        initial = current_val if current_val and os.path.exists(current_val) else os.getcwd()
        selected_dir = filedialog.askdirectory(parent=root, initialdir=initial, title="Chọn thư mục lưu file")
        root.destroy()
        return selected_dir if selected_dir else current_val
    except Exception as e:
        print(f"[browse_folder] Không có giao diện đồ họa GUI (headless/Colab): {e}")
        return current_val

# Thuộc tính HTML gắn trực tiếp lên ô nhập (Gradio ≥ 4) — tắt spellcheck trình duyệt cho tiếng Việt.
TTS_TEXT_HTML_ATTRS = cast(
    InputHTMLAttributes,
    {
        "spellcheck": False,
        "autocorrect": "off",
        "autocapitalize": "off",
        "autocomplete": "off",
        "lang": "vi",
    },
)

# ── Build UI ───────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="🎙️ OmniVoice Studio Pro - AI Voice Cloning",
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate", neutral_hue="slate"),
    css=CSS,
    js=APP_INIT_JS
) as demo:

    gr.HTML(f"""
        <div class="studio-header-card" style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
            <div>
                <div style="display: flex; align-items: center; gap: 0.75rem;">
                    <span style="font-size: 2.2rem;">🎙️</span>
                    <h1 class="studio-title-text" style="display: inline-block;">OMNIVOICE STUDIO</h1>
                    <span style="font-size: 0.75rem; font-weight: 800; background: linear-gradient(135deg, #6366f1, #06b6d4); color: white; padding: 3px 9px; border-radius: 6px; letter-spacing: 0.05em;">PRO</span>
                </div>
                <p style="color: #94a3b8; margin: 0.4rem 0 0 0; font-size: 0.88rem; font-weight: 500;">
                    High-Performance Neural Text-to-Speech & Voice Cloning Workstation
                </p>
            </div>
            <div style="display: flex; gap: 0.6rem; flex-wrap: wrap; align-items: center;">
                <span class="studio-pill"><span class="pulse-dot-green"></span> Engine: {DEVICE.upper()} (OmniVoice)</span>
                <span class="studio-pill">⚡ FlashAttention-2 + TF32 + BF16</span>
                <span class="studio-pill">🔊 24kHz HD Audio</span>
            </div>
        </div>
    """)

    with gr.Row(equal_height=False, elem_id="main-row"):
        # Left Column: Voice & Audio Settings Deck
        with gr.Column(scale=1, elem_classes=["card"]):
            gr.HTML('<div class="studio-section-title">🎤 1. Reference Voice Selection</div>')

            gr.HTML('''<div class="studio-tip-banner">
                <span style="color: #38bdf8; font-weight: 700;">💡 Studio Tip:</span> Để OmniVoice nhân bản chính xác nhất, đặt file <code style="background: rgba(255,255,255,0.12); padding: 1px 5px; border-radius: 4px;">.wav</code> và <code style="background: rgba(255,255,255,0.12); padding: 1px 5px; border-radius: 4px;">.txt</code> cùng tên vào thư mục <code style="background: rgba(255,255,255,0.12); padding: 1px 5px; border-radius: 4px;">wavs/</code>.
            </div>''')      
            
            wav_files = list_voices()
            default_voice = get_default_voice(wav_files)
            if wav_files:
                ref_dropdown = gr.Dropdown(
                    choices=[(Path(f).stem, f) for f in wav_files],
                    label="Select Reference Voice",
                    value=default_voice,
                )
            else:
                ref_dropdown = gr.Dropdown(choices=[], label="No voices in wavs/")

            ref_audio = gr.Audio(
                label="Or Upload / Record Reference Audio",
                type="filepath",
                value=default_voice,
                sources=["upload", "microphone"],
            )

            gr.HTML('<div class="studio-section-title" style="margin-top: 1.25rem;">🎨 2. Vocal Performance Tuning</div>')

            language = gr.Radio(
                choices=[("🇻🇳 Tiếng Việt", "vi"), ("🇺🇸 English", "en")],
                value="vi", label="Language"
            )

            with gr.Row():
                ai_speed = gr.Slider(
                    minimum=0.7,
                    maximum=1.5,
                    step=0.05,
                    value=1.0,
                    label="🏎️ Speech Speed",
                    info="Tốc độ nói. 1.0=Chuẩn, >1.0=Nhanh, <1.0=Chậm.",
                )
                ui_pitch_shift = gr.Slider(
                    minimum=0.5,
                    maximum=2.0,
                    step=0.05,
                    value=1.0,
                    label="🎵 Pitch Shift",
                    info="Cao độ giọng. 1.0=Chuẩn, >1=Trầm, <1=Bổng.",
                )

            gr.HTML('<div class="studio-section-title" style="margin-top: 1.25rem;">📂 3. Export Directory Configuration</div>')
            with gr.Row():
                folder_input = gr.Textbox(
                    label="Download Folder Path",
                    placeholder="Nhập đường dẫn thư mục lưu...",
                    value=load_path(),
                    scale=3
                )
                browse_btn = gr.Button("📂 Browse", scale=1, elem_classes=["studio-btn-secondary"])
                save_btn = gr.Button("💾 Save Path", scale=1, elem_classes=["studio-btn-secondary"])

        # Right Column: Script Editor & Studio Deck
        with gr.Column(scale=1, elem_classes=["card"]):
            gr.HTML('<div class="studio-section-title">📝 4. Studio Script Editor</div>')

            text_input = gr.Textbox(
                label="Text Script to Synthesize",
                placeholder="Nhập văn bản Kịch bản / Lời thoại cần đọc tại đây...",
                lines=7,
                elem_id="main-text-input",
                html_attributes=TTS_TEXT_HTML_ATTRS,
            )

            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear Script", variant="secondary", size="sm", elem_classes=["studio-btn-secondary"])
                sample1_btn = gr.Button("✨ Sample 1 (Giới Thiệu)", variant="secondary", size="sm", elem_classes=["studio-btn-secondary"])
                sample2_btn = gr.Button("✨ Sample 2 (Truyện Kể)", variant="secondary", size="sm", elem_classes=["studio-btn-secondary"])

            gr.HTML('<div class="studio-section-title" style="margin-top: 1.25rem;">🚀 5. Neural Synthesis Execution</div>')

            with gr.Row():
                generate_btn = gr.Button("⚡ SYNTHESIZE SPEECH + SRT", variant="primary", size="lg", elem_classes=["generate-btn"])
                generate_speech_only_btn = gr.Button("🔊 SYNTHESIZE SPEECH ONLY", variant="secondary", size="lg", elem_classes=["studio-btn-secondary"])

            gr.HTML('<div class="studio-section-title" style="margin-top: 1.25rem;">🔈 6. Studio Audio Deck</div>')

            with gr.Row():
                output_audio = gr.Audio(label="Generated Speech Waveform", type="numpy", scale=2, interactive=False)
                status_text = gr.Textbox(label="Telemetry Status Console", lines=3, scale=1, elem_classes=["status-console"])

            save_audio_btn = gr.Button("💾 Save Audio & Subtitle Files to Target Directory", variant="secondary", elem_classes=["studio-btn-secondary"])

            with gr.Row():
                saved_file = gr.File(label="Saved Audio File (.WAV)", interactive=False)
                srt_file = gr.File(label="Subtitle File (.SRT)", interactive=False, visible=True)

    clear_btn.click(fn=lambda: "", outputs=[text_input])
    sample1_btn.click(
        fn=lambda: "Chào mừng bạn đến với OmniVoice Studio Pro. Hệ thống nhân bản giọng nói AI thế hệ mới giúp bạn tạo ra những bản thu âm tự nhiên, truyền cảm và chuyên nghiệp nhất.",
        outputs=[text_input]
    )
    sample2_btn.click(
        fn=lambda: "Hôm nay thời tiết thật là đẹp, trời xanh mây trắng rất thích hợp để ra ngoài dạo chơi và thưởng thức một tách cà phê thơm ngon cùng bạn bè.",
        outputs=[text_input]
    )

    ref_dropdown.change(fn=lambda x: gr.update(value=x), inputs=[ref_dropdown], outputs=[ref_audio])
    ref_audio.clear(fn=lambda: None, outputs=[ref_dropdown])

    def _generate_core(data, progress=None):
        try:
            omni_model = get_omni_model()
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, f"❌ Omni load error: {str(e)}"

        ref_audio_temp_path = data[ref_audio]
        audio_filename = Path(ref_audio_temp_path).stem

        wavs_dir = get_wavs_dir()
        ref_audio_path = wavs_dir / f"{audio_filename}.wav"
        ref_text_path = wavs_dir / f"{audio_filename}.txt"

        if not ref_audio_path.exists():
            ref_audio_path = Path(ref_audio_temp_path)

        ref_text = None
        if ref_text_path.exists():
            try:
                with open(ref_text_path, "r", encoding="utf-8") as f:
                    ref_text = f.read().strip()
            except Exception:
                ref_text = None

        audio_out, status, srtFileResult = generate_speech_omni(
            omni=omni_model,
            text=data[text_input],
            language=data[language],
            reference_audio=str(ref_audio_path),
            ref_text=ref_text,
            speed=data[ai_speed],
            pitch_shift=data[ui_pitch_shift],
            progress=progress,
        )
        return audio_out, status, srtFileResult

    def generate_speech_and_srt_fn(data, progress=gr.Progress()):
        progress(0.0, desc="Khởi tạo mô hình OmniVoice và tải giọng nói mẫu...")
        audio_out, status, srt_path = _generate_core(data, progress=progress)
        if audio_out is None:
            return None, status, None, None
        
        progress(0.9, desc="Đang tự động lưu file âm thanh và phụ đề SRT...")
        folder_path = data[folder_input]
        text = data[text_input]
        save_status, saved_audio_path = save_generated_audio_and_srt(
            audio_out, text, folder_path, srt_path
        )
        status = f"{status}\n{save_status}"
        progress(1.0, desc="Hoàn thành!")
        return audio_out, status, srt_path, saved_audio_path

    def generate_speech_only_fn(data, progress=gr.Progress()):
        progress(0.0, desc="Khởi tạo mô hình OmniVoice và tải giọng nói mẫu...")
        audio_out, status, srt_path = _generate_core(data, progress=progress)
        if audio_out is None:
            return None, status, None, None
        
        progress(0.9, desc="Đang tự động lưu file âm thanh...")
        folder_path = data[folder_input]
        text = data[text_input]
        save_status, saved_audio_path = save_generated_audio_and_srt(
            audio_out, text, folder_path, None
        )
        status = f"{status}\n{save_status}"
        progress(1.0, desc="Hoàn thành!")
        return audio_out, status, None, saved_audio_path

    all_inputs = {
        text_input, language, ref_audio, ai_speed, ui_pitch_shift, folder_input
    }

    generate_btn.click(
        fn=generate_speech_and_srt_fn,
        inputs=all_inputs,
        outputs=[output_audio, status_text, srt_file, saved_file]
    )

    generate_speech_only_btn.click(
        fn=generate_speech_only_fn,
        inputs=all_inputs,
        outputs=[output_audio, status_text, srt_file, saved_file]
    )

    browse_btn.click(fn=browse_folder, inputs=folder_input, outputs=folder_input)
    save_btn.click(fn=save_path, inputs=folder_input, outputs=status_text)

    save_audio_btn.click(
        fn=save_generated_audio_and_srt,
        inputs=[output_audio, text_input, folder_input, srt_file],
        outputs=[status_text, saved_file],
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="OmniVoice Studio Pro Web UI")
    parser.add_argument("--share", action="store_true", default=None, help="Bật link chia sẻ Gradio live (tự động bật trên Colab)")
    parser.add_argument("--port", type=int, default=7860, help="Port chạy server (mặc định 7860)")
    args, _ = parser.parse_known_args()

    # Nhận diện môi trường Google Colab
    is_colab = "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ or os.path.exists("/content")
    if not is_colab:
        try:
            import importlib.util
            if importlib.util.find_spec("google.colab") is not None:
                is_colab = True
        except Exception:
            pass

    share_mode = True if args.share is True or (args.share is None and is_colab) else False
    inbrowser_mode = not is_colab and not share_mode

    print(f"🚀 Khởi chạy Web UI: server=0.0.0.0, port={args.port}, share={share_mode}, in_colab={is_colab}")
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=share_mode, inbrowser=inbrowser_mode)

