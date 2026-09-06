"""
Minimal OmniVoice wrapper for Gradio app integration.
"""
from __future__ import annotations

# Set HF Hub env vars BEFORE importing transformers to disable warnings
import os
import re
import math
from pedalboard import Pedalboard as PB, PitchShift

# Disable telemetry: Prevent Hugging Face from sending usage statistics/analytics
# Điều này tránh các request ngầm đến HF Hub để báo cáo dữ liệu sử dụng
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# Force offline mode: Không gọi API đến HF Hub, chỉ dùng model local
# Điều này tránh warning "unauthenticated requests" vì không còn request nào được gửi đi
os.environ["HF_HUB_OFFLINE"] = "1"

# Set dummy token để tránh warning "unauthenticated requests"
# Vì đang ở offline mode, token này sẽ không được sử dụng cho bất kỳ request nào
# nhưng sẽ làm hài lòng auth check của huggingface_hub
os.environ["HF_TOKEN"] = "dummy"

# Disable symlink warning: Tránh warning về việc Windows không hỗ trợ symlinks tốt
# (thường xuất hiện khi HF Hub cố tạo symlink cho cache files)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Suppress transformers logging
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

from pathlib import Path
from typing import Any, Optional, cast
import sys
import gc
import warnings
import tempfile, os
import numpy as np
import torch

try:
    from .omnivoice_support.ttsOmni_Config import inferWithModelOmni, get_voice_clone_prompt
except ImportError:
    from OmniVoice.omnivoice_inference.omnivoice_support.ttsOmni_Config import (  # type: ignore
        inferWithModelOmni,
        get_voice_clone_prompt,
    )

# Thêm thư mục gốc của project vào sys.path để import được module 'general'
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from general.general_tool_audio import (  # type: ignore
    SEGMENT_TEXT,
    get_reference_sound,
    segment_text,
    fix_silent_and_speed_audio,
    clearText,
    create_srt_file
)
from general.noise_detect_VAD import vad_trim  # type: ignore

def _import_omnivoice_class():
    try:
        from omnivoice.models.omnivoice import OmniVoice as OmniVoiceClass
        return OmniVoiceClass
    except ModuleNotFoundError:
        # Fallback when OmniVoice is present as local source (repo checkout).
        local_omnivoice_root = Path(__file__).resolve().parents[1]
        local_omnivoice_root_str = str(local_omnivoice_root)
        if local_omnivoice_root_str not in sys.path:
            sys.path.insert(0, local_omnivoice_root_str)
        from omnivoice.models.omnivoice import OmniVoice as OmniVoiceClass
        return OmniVoiceClass


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Omni:
    """Lazy-loaded OmniVoice model wrapper."""

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.device = device or _best_device()
        self.model_path = self._resolve_model_path(model_path)
        self.model: Optional[Any] = None

    @staticmethod
    def _validate_local_model_dir(model_dir: Path) -> None:
        required_files = [
            model_dir / "config.json",
        ]
        weight_candidates = [
            model_dir / "model.safetensors",
            model_dir / "pytorch_model.bin",
            model_dir / "model.safetensors.index.json",
            model_dir / "pytorch_model.bin.index.json",
        ]

        missing = [p for p in required_files if not p.exists()]
        has_any_weight = any(p.exists() for p in weight_candidates)
        if not has_any_weight:
            # Show the expected weight file names for clarity.
            missing.extend(weight_candidates)

        if missing:
            print("❌ Thiếu file quan trọng trong model Omni local. Không thể load:")
            for p in missing:
                print(f"- {p.as_posix()}")
            raise FileNotFoundError(
                f"Omni local model incomplete at '{model_dir.as_posix()}'. Missing required files."
            )

    @staticmethod
    def _resolve_model_path(model_path: Optional[str]) -> str:
        if model_path:
            return model_path

        candidate = Path("OmniVoice/modelOmniLocal")
        if candidate.exists():
            if not candidate.is_dir():
                print("❌ Model Omni local path tồn tại nhưng không phải thư mục:")
                print(f"- {candidate.as_posix()}")
                raise NotADirectoryError(candidate.as_posix())

            Omni._validate_local_model_dir(candidate)
            print("🏠 Model Omni local có tồn tại\n")
            return str(candidate)

        print("Model Omni local KHÔNG tồn tại")
        return "k2-fsa/OmniVoice"

    def loadModelOmni(self) -> Any:
        if self.model is None:
            # Sử dụng Float16 trên GPU (T4 / V100 / RTX) để tăng tốc vượt trội và tiết kiệm VRAM
            if torch.cuda.is_available():
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                dtype = torch.float32
            omni_voice_cls = _import_omnivoice_class()
            model = cast(Any, omni_voice_cls.from_pretrained(
                self.model_path,
                dtype=dtype,
            ))
            model = model.to(self.device)
            self.model = model
            print(f"📝 Model loaded in dtype: {dtype} | ASR chỉ load khi cần xài\n")
        return cast(Any, self.model)

    def loadOmniFromUI(self):
        return self.loadModelOmni()

    _inferWithModelOmni = inferWithModelOmni

    @property
    def sampling_rate(self) -> int:
        if not hasattr(self, '_sampling_rate'): # chưa có trong cache thì load model để lấy sampling_rate
            model = cast(Any, self.loadModelOmni())
            self._sampling_rate = cast(int, model.sampling_rate) # đã cache
        return self._sampling_rate


def split_text_into_chunks(text: str, max_chars: int = 240) -> list[str]:
    """Gom các câu thành các đoạn đọc hợp lý (mặc định tối đa 240 ký tự).
    Giữ nguyên ngữ điệu tự nhiên giữa các dấu phẩy, giảm tối đa số lần gọi model."""
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    # Cắt theo dấu chấm, chấm than, hỏi chấm, chấm lửng hoặc xuống dòng
    sentences = re.split(r"(?<=[.!?…;\n])\s+", text)
    chunks: list[str] = []
    cur = ""
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        candidate = (cur + " " + sent).strip() if cur else sent
        if len(candidate) <= max_chars:
            cur = candidate
        else:
            if cur:
                chunks.append(cur)
                cur = ""
            if len(sent) <= max_chars:
                cur = sent
            else:
                # Nếu 1 câu quá dài vượt quá max_chars, cắt theo dấu phẩy hoặc từ
                sub_parts = re.split(r"(?<=[,，:：])\s+", sent)
                for part in sub_parts:
                    part = part.strip()
                    if not part:
                        continue
                    c_part = (cur + " " + part).strip() if cur else part
                    if len(c_part) <= max_chars:
                        cur = c_part
                    else:
                        if cur:
                            chunks.append(cur)
                        cur = part
    if cur:
        chunks.append(cur)
    return chunks


def generate_speech_omni(
    omni: Omni,
    text: str,
    language: str = "vi",
    reference_audio: Optional[str] = None,
    ref_text: Optional[str] = None,
    speed: float = 1.0,
    pitch_shift: float = 1.0,            # F0 scaling pitch: 0.5~2.0 (1.0=bình thường)
    num_step: int = 16,
    max_chars: int = 240,
    progress_callback = None,
    progress = None,
):
    if not (text or "").strip():
        return None, "❌ Please enter some text", None
    if not reference_audio:
        ref_path = get_reference_sound()
        if ref_path is None:
            return None, "❌ No reference audio! Add .wav files to wavs/ folder", None
        reference_audio = str(ref_path)

    # ── Preprocess text ────────────────────────────────────────────────────
    text = clearText(text, language)

    # Gom câu thành các chunk đọc hợp lý
    chunks = split_text_into_chunks(text, max_chars=max_chars)
    if not chunks:
        chunks = [text]

    total_chunks = len(chunks)
    print(f"\n📝 Text chunked into {total_chunks} blocks (max_chars={max_chars}):", flush=True)
    for idx, c in enumerate(chunks):
        print(f"   [{idx+1}/{total_chunks}] 🗣 [{c[:60]}{'…' if len(c)>60 else ''}] ({len(c)} chars)")

    # Tạo voice clone prompt 1 LẦN DUY NHẤT để tái sử dụng
    model = omni.loadModelOmni()
    voice_clone_prompt = get_voice_clone_prompt(
        reference_audio=reference_audio,
        ref_text=ref_text,
        model=model,
        preprocess_prompt=True,
        language=language,
    )

    result: Optional[np.ndarray] = None
    arrSrt: list[dict] = []
    current_time: float = 0.0

    print(f"\n🚩 Bắt đầu inference audio OmniVoice: num_step={num_step}, speed={speed}, pitch={pitch_shift}", flush=True)

    pause_between_chunks_ms = 200
    silence_inter = np.zeros(int(omni.sampling_rate * pause_between_chunks_ms / 1000), dtype=np.float32)

    for chunk_idx, chunk in enumerate(chunks):
        preview = chunk[:50] + ("…" if len(chunk) > 50 else "")
        if progress_callback:
            progress_callback(chunk_idx + 1, total_chunks, preview)
        if progress:
            progress((chunk_idx) / total_chunks, desc=f"Đang sinh giọng nói - Đoạn {chunk_idx + 1}/{total_chunks}: {preview}")

        print(f"\n===================================================")
        print(f"  🔊📢 [Chunk {chunk_idx + 1}/{total_chunks}]: {chunk}\n", flush=True)

        audios = omni._inferWithModelOmni(
            text=chunk.strip(),
            reference_audio=reference_audio,
            ref_text=ref_text,
            language=language,
            speed=speed,
            num_step=num_step,
            voice_clone_prompt=voice_clone_prompt,
        )

        getFirstAudio = audios[0]
        getFirstAudio = vad_trim(getFirstAudio, omni.sampling_rate, margin_s=0.05)
        audio_np = fix_silent_and_speed_audio(
            getFirstAudio, omni.sampling_rate,
            threshold_ms=50,
            silence_threshold_db=-45
        )

        # Pitch shift post-processing
        if pitch_shift != 1.0:
            n_semitones = 12.0 * math.log2(max(0.5, min(2.0, float(pitch_shift))))
            try:
                pitch_board = PB([PitchShift(semitones=n_semitones)])
                audio_2d = audio_np.reshape(1, -1).astype(np.float32)
                audio_np = pitch_board(audio_2d, omni.sampling_rate).flatten()
            except Exception as e:
                print(f"⚠️ pitch_shift (pedalboard) failed: {e}")

        chunk_duration = len(audio_np) / omni.sampling_rate
        if chunk_duration > 0:
            start_time = current_time
            end_time = current_time + chunk_duration

            arrSrt.append({
                "startTime": start_time,
                "endTime": end_time,
                "text": chunk
            })

            piece = audio_np.astype(np.float32)
            if result is None:
                result = piece
            else:
                result = np.concatenate([result, silence_inter, piece])
                current_time += (pause_between_chunks_ms / 1000.0)

            current_time += chunk_duration
            print(f"  🎵 Audio chunk generated: {chunk_duration:.2f}s | timeline {start_time:.2f}s - {end_time:.2f}s", flush=True)

        # Đồng bộ CUDA và dọn dẹp bộ nhớ sau mỗi chunk để tránh OOM / sập Kernel Colab
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()

    if result is None:
        return None, "❌ Không tạo được âm thanh từ text", None

    # Khoảng lặng đuôi 250ms
    trailing_samples = int(0.25 * omni.sampling_rate)
    silence_end = np.zeros(trailing_samples, dtype=np.float32)
    result = np.concatenate([result, silence_end])

    duration = len(result) / omni.sampling_rate
    status = f"✅ Hoàn thành! | {duration:.2f}s | {language.upper()} ({total_chunks} đoạn)"

    gradio_temp = os.environ.get("GRADIO_TEMP_DIR", tempfile.gettempdir())
    srt_temp_path = os.path.join(gradio_temp, f"omni_srt_{hash(text) % 1000000}.srt")
    create_srt_file(arrSrt, srt_temp_path)

    print(f"✅ Created SRT file: {srt_temp_path}", flush=True)
    print(f"✅ Done inference OmniVoice | duration={duration:.2f}s | total_chunks={total_chunks}\n", flush=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    if progress:
        progress(1.0, desc="Hoàn thành!")

    return (omni.sampling_rate, result), status, srt_temp_path