"""
OmniVoice Studio Pro - High-Performance Gradio Web Interface
"""
import os
import sys
import time
import html
import shutil
import warnings
import tempfile
import threading
from pathlib import Path
from typing import cast

# Tắt cảnh báo không cần thiết từ thư viện
warnings.filterwarnings('ignore')
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
import gradio as gr
from gradio.components.textbox import InputHTMLAttributes

os.environ["GRADIO_TEMP_DIR"] = tempfile.gettempdir() + "/my_gradio_tmp"
os.makedirs(os.environ["GRADIO_TEMP_DIR"], exist_ok=True)

# Clear temp folder on startup
try:
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
    load_path, save_generated_audio_and_srt,
)

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"Device: {DEVICE}")

configure_device(DEVICE)

# -----------------------------------------------------------------------------
# THREAD-SAFE STATE & PROGRESS TRACKER
# -----------------------------------------------------------------------------
STATE_LOCK = threading.Lock()
STATE = {
    "job_running": False,
    "job_pct": 0,
    "job_msg": "Hệ thống sẵn sàng",
    "job_detail": "Nhập nội dung cần đọc, chọn giọng tham chiếu rồi bấm TẠO GIỌNG NÓI.",
    "job_started": 0.0,
    "status_class": "ready",
}

def state_update(**kwargs):
    with STATE_LOCK:
        STATE.update(kwargs)

def snapshot_state():
    with STATE_LOCK:
        return dict(STATE)

def fmt_elapsed(seconds):
    seconds = max(0, int(seconds))
    m, s = divmod(seconds, 60)
    return f"{m:02d}:{s:02d}"

def status_html():
    s = snapshot_state()
    cls = s.get("status_class", "ready")
    if s["job_running"]:
        pct = int(max(1, min(99, s["job_pct"])))
        elapsed = fmt_elapsed(time.time() - s["job_started"])
        msg = s["job_msg"] or "Đang tổng hợp giọng nói…"
        detail = (s["job_detail"] + f" · Đã chạy {elapsed}").strip(" ·")
        icon = "⚡"
    elif cls == "error":
        pct = 100
        msg = s["job_msg"] or "Lỗi xử lý"
        detail = s["job_detail"]
        icon = "❌"
    else:
        pct = 100 if s["job_pct"] == 100 else 0
        msg = s["job_msg"] or "Hệ thống sẵn sàng"
        detail = s["job_detail"] or "Nhập văn bản và bấm TẠO GIỌNG NÓI"
        icon = "✓"

    msg_safe = html.escape(str(msg))
    detail_safe = html.escape(str(detail))
    return f"""
    <div class='studio-status-card studio-status-{cls}'>
      <div class='studio-status-top'>
        <div><span class='studio-status-icon'>{icon}</span><b>{msg_safe}</b></div>
        <div class='studio-pct'>{pct}%</div>
      </div>
      <div class='studio-progress-track'><div class='studio-progress-fill' style='width:{pct}%'></div></div>
      <div class='studio-status-detail'>{detail_safe}</div>
    </div>
    """

# Thuộc tính HTML tắt spellcheck trình duyệt cho tiếng Việt
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
                <span class="studio-pill"><span class="pulse-dot-green"></span> Engine: {DEVICE.upper()} (OmniVoice FP16)</span>
                <span class="studio-pill">⚡ Chunking Pipeline</span>
                <span class="studio-pill">🔊 24kHz HD Audio</span>
            </div>
        </div>
    """)

    with gr.Row(equal_height=False, elem_id="main-row"):
        # Left Column: Voice & Audio Settings Deck
        with gr.Column(scale=1, elem_classes=["card"]):
            gr.HTML('<div class="studio-section-title">🎤 1. Reference Voice Selection</div>')

            gr.HTML('''<div class="studio-tip-banner">
                <span style="color: #38bdf8; font-weight: 700;">💡 Studio Tip:</span> Đặt file <code style="background: rgba(255,255,255,0.12); padding: 1px 5px; border-radius: 4px;">.wav</code> và <code style="background: rgba(255,255,255,0.12); padding: 1px 5px; border-radius: 4px;">.txt</code> cùng tên vào thư mục <code style="background: rgba(255,255,255,0.12); padding: 1px 5px; border-radius: 4px;">wavs/</code> để nhân bản giọng chính xác nhất.
            </div>''')      
            
            wav_files = list_voices()
            default_voice = get_default_voice(wav_files)
            if wav_files:
                ref_dropdown = gr.Dropdown(
                    choices=[(Path(f).stem, f) for f in wav_files],
                    label="Select Reference Voice (từ thư mục wavs)",
                    value=default_voice,
                )
            else:
                ref_dropdown = gr.Dropdown(choices=[], label="No voices in wavs/")

            ref_audio = gr.Audio(
                label="Âm thanh giọng mẫu (Upload hoặc Thu âm trực tiếp)",
                type="filepath",
                value=default_voice,
                sources=["upload", "microphone"],
            )

            gr.HTML('<div class="studio-section-title" style="margin-top: 1.25rem;">🎨 2. Vocal Performance Tuning</div>')

            language = gr.Radio(
                choices=[("🇻🇳 Tiếng Việt", "vi"), ("🇺🇸 English", "en")],
                value="vi", label="Ngôn ngữ"
            )

            with gr.Row():
                ai_speed = gr.Slider(
                    minimum=0.7,
                    maximum=1.5,
                    step=0.05,
                    value=1.0,
                    label="🏎️ Tốc độ nói (Speed)",
                    info="1.0=Chuẩn, >1.0=Nhanh, <1.0=Chậm.",
                )
                ui_pitch_shift = gr.Slider(
                    minimum=0.5,
                    maximum=2.0,
                    step=0.05,
                    value=1.0,
                    label="🎵 Cao độ giọng (Pitch)",
                    info="1.0=Chuẩn, >1=Trầm, <1=Bổng.",
                )

            with gr.Accordion("⚙️ Cài đặt tối ưu nâng cao", open=False):
                ui_num_step = gr.Slider(
                    minimum=8,
                    maximum=32,
                    step=1,
                    value=16,
                    label="Số bước khử nhiễu (num_step)",
                    info="16 bước: Rất nhanh & tối ưu GPU T4 (khuyên dùng). 24-32 bước: Chi tiết hơn nhưng tốn thời gian hơn.",
                )
                ui_max_chars = gr.Slider(
                    minimum=140,
                    maximum=360,
                    step=20,
                    value=240,
                    label="Độ dài gom đoạn (max_chars)",
                    info="Gom các câu ngắn thành đoạn đọc liền mạch tự nhiên (mặc định 240 ký tự).",
                )

        # Right Column: Script Editor & Studio Deck
        with gr.Column(scale=1, elem_classes=["card"]):
            gr.HTML('<div class="studio-section-title">📝 3. Studio Script Editor</div>')

            text_input = gr.Textbox(
                label="Văn bản cần đọc (Script)",
                placeholder="Nhập văn bản Kịch bản / Lời thoại cần đọc tại đây...",
                lines=8,
                elem_id="main-text-input",
                html_attributes=TTS_TEXT_HTML_ATTRS,
            )

            with gr.Row():
                clear_btn = gr.Button("🗑️ Xóa kịch bản", variant="secondary", size="sm", elem_classes=["studio-btn-secondary"])
                sample1_btn = gr.Button("✨ Sample 1 (Giới Thiệu)", variant="secondary", size="sm", elem_classes=["studio-btn-secondary"])
                sample2_btn = gr.Button("✨ Sample 2 (Truyện Kể)", variant="secondary", size="sm", elem_classes=["studio-btn-secondary"])

            gr.HTML('<div class="studio-section-title" style="margin-top: 1.25rem;">🚀 4. Neural Synthesis Execution</div>')

            generate_btn = gr.Button("⚡ TẠO GIỌNG NÓI & PHỤ ĐỀ (SYNTHESIZE SPEECH)", variant="primary", size="lg", elem_classes=["generate-btn"])

            status_box = gr.HTML(status_html())

            gr.HTML('<div class="studio-section-title" style="margin-top: 1.25rem;">🔈 5. Studio Audio Deck</div>')

            output_audio = gr.Audio(label="Bản thu âm đã tạo (Generated Speech Waveform)", type="numpy", interactive=False)
            status_text = gr.Textbox(label="Telemetry Status Console", lines=2, elem_classes=["status-console"])

            with gr.Row():
                saved_file = gr.File(label="Tải file âm thanh (.WAV)", interactive=False)
                srt_file = gr.File(label="Tải file phụ đề (.SRT)", interactive=False, visible=True)

    # Nút bấm mẫu
    clear_btn.click(fn=lambda: "", outputs=[text_input])
    sample1_btn.click(
        fn=lambda: "Chào mừng bạn đến với OmniVoice Studio Pro. Hệ thống nhân bản giọng nói AI thế hệ mới giúp bạn tạo ra những bản thu âm tự nhiên, truyền cảm và chuyên nghiệp nhất.",
        outputs=[text_input]
    )
    sample2_btn.click(
        fn=lambda: "Hôm nay thời tiết thật là đẹp, trời xanh mây trắng rất thích hợp để ra ngoài dạo chơi và thưởng thức một tách cà phê thơm ngon cùng bạn bè.",
        outputs=[text_input]
    )

    # Khi chọn voice trong dropdown -> cập nhật vào ref_audio (không gán ngược để tránh circular loop)
    ref_dropdown.change(fn=lambda x: x, inputs=[ref_dropdown], outputs=[ref_audio])

    # Timer polling trạng thái mỗi 0.8 giây
    timer = gr.Timer(0.8)
    timer.tick(status_html, outputs=status_box, queue=False, show_progress="hidden")

    def generate_speech_fn(data):
        started = time.time()
        text = data[text_input]
        ref_temp_path = data[ref_audio]

        if not (text or "").strip():
            state_update(
                job_running=False,
                job_pct=100,
                job_msg="Chưa nhập văn bản!",
                job_detail="Vui lòng nhập nội dung cần đọc vào ô Script.",
                status_class="error",
            )
            return None, "❌ Vui lòng nhập văn bản cần đọc!", None, None

        state_update(
            job_running=True,
            job_pct=5,
            job_msg="Khởi tạo mô hình & chuẩn bị giọng mẫu…",
            job_detail="Đang nạp mô hình OmniVoice",
            job_started=started,
            status_class="running",
        )

        try:
            omni_model = get_omni_model()
        except Exception as e:
            state_update(
                job_running=False,
                job_pct=100,
                job_msg="Lỗi nạp mô hình",
                job_detail=str(e),
                status_class="error",
            )
            return None, f"❌ Omni load error: {str(e)}", None, None

        audio_filename = Path(ref_temp_path).stem if ref_temp_path else "default"
        wavs_dir = get_wavs_dir()
        ref_audio_path = wavs_dir / f"{audio_filename}.wav"
        ref_text_path = wavs_dir / f"{audio_filename}.txt"

        if not ref_audio_path.exists():
            ref_audio_path = Path(ref_temp_path) if ref_temp_path else None

        ref_text = None
        if ref_text_path.exists():
            try:
                with open(ref_text_path, "r", encoding="utf-8") as f:
                    ref_text = f.read().strip()
            except Exception:
                ref_text = None

        def progress_cb(cur_chunk, total_chunks, preview):
            pct = int(10 + 80 * ((cur_chunk - 1) / max(1, total_chunks)))
            state_update(
                job_pct=pct,
                job_msg=f"Đang tạo giọng · Đoạn {cur_chunk}/{total_chunks}",
                job_detail=f"Nội dung: {preview}",
                status_class="running",
            )

        audio_out, status, srt_path = generate_speech_omni(
            omni=omni_model,
            text=text,
            language=data[language],
            reference_audio=str(ref_audio_path) if ref_audio_path else None,
            ref_text=ref_text,
            speed=data[ai_speed],
            pitch_shift=data[ui_pitch_shift],
            num_step=int(data[ui_num_step]),
            max_chars=int(data[ui_max_chars]),
            progress_callback=progress_cb,
        )

        if audio_out is None:
            state_update(
                job_running=False,
                job_pct=100,
                job_msg="Không tạo được âm thanh",
                job_detail=status,
                status_class="error",
            )
            return None, status, None, None

        state_update(
            job_pct=95,
            job_msg="Đang hoàn tất đóng gói file âm thanh & phụ đề…",
            job_detail="Lưu file tạm vào Gradio deck",
            status_class="running",
        )

        save_status, saved_audio_path = save_generated_audio_and_srt(
            audio_out, text, "", srt_path
        )
        status = f"{status}\n{save_status}"

        elapsed_total = time.time() - started
        state_update(
            job_running=False,
            job_pct=100,
            job_msg="✓ Hoàn thành tạo giọng nói!",
            job_detail=f"Thời gian xử lý: {fmt_elapsed(elapsed_total)} · File đã sẵn sàng tải về bên dưới.",
            status_class="ready",
        )

        return audio_out, status, srt_path, saved_audio_path

    all_inputs = {
        text_input, language, ref_audio, ai_speed, ui_pitch_shift, ui_num_step, ui_max_chars
    }

    generate_btn.click(
        fn=generate_speech_fn,
        inputs=all_inputs,
        outputs=[output_audio, status_text, srt_file, saved_file]
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
