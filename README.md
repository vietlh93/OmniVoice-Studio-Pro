# 🎙️ OmniVoice Studio Pro

**High-Performance Neural Text-to-Speech & Voice Cloning Workstation**  
Hệ thống nhân bản giọng nói AI thế hệ mới, hỗ trợ tiếng Việt và tiếng Anh, tối ưu độ tự nhiên và cảm xúc.

---

## ✨ Tính năng nổi bật

- **Voice Cloning độ chính xác cao**: Nhân bản giọng nói bất kỳ từ file mẫu audio chỉ từ 3-10 giây với mô hình **OmniVoice Tiếng Việt (KhanhTTS)**.
- **Tự động tạo phụ đề SRT**: Khi tạo giọng nói, hệ thống tự động đồng bộ và sinh file phụ đề `.srt` chính xác theo từng câu nói.
- **ASR Chunkformer Tiếng Việt**: Tích hợp mô hình Chunkformer ASR chuyên sâu cho tiếng Việt để tự động nhận diện và bóc băng lời thoại từ file giọng mẫu.
- **Higgs Audio V2 Tokenizer**: Sử dụng bộ mã hóa âm thanh hiện đại, giữ nguyên độ chi tiết và âm sắc tự nhiên của giọng nói.
- **Tùy biến cao độ & tốc độ**: Cho phép điều chỉnh **Speed** (tốc độ nói) và **Pitch Shift** (cao độ giọng bổng/trầm) trực quan trên giao diện.
- **Silero VAD lọc âm thông minh**: Tự động phát hiện khoảng lặng, ngắt câu theo ngữ nghĩa tự nhiên, hạn chế tối đa tình trạng nuốt chữ hoặc hụt hơi.
- **Hỗ trợ 2 nền tảng**:
  - **Chạy cục bộ trên Windows**: Khởi động 1-click bằng `runApp.bat`.
  - **Chạy trên Google Colab**: Tự động toàn diện qua `OmniVoice_Studio_Colab.ipynb` (ẩn log rác, có thanh tiến trình trực quan).

---

## 📁 Cấu trúc dự án

```
OmniVoice-Studio-Pro/
├── app.py                         # Giao diện chính Gradio Web UI
├── runApp.bat                     # Script khởi chạy nhanh trên Windows
├── OmniVoice_Studio_Colab.ipynb   # Notebook khởi chạy trên Google Colab
├── OmniVoiceStudioPro.spec        # Cấu hình đóng gói ứng dụng PyInstaller
├── general/                       # Thư viện hỗ trợ & cấu hình
│   ├── general_tool_audio.py      # Xử lý text và chuẩn hóa âm thanh
│   ├── noise_detect_VAD.py        # Tích hợp Silero VAD
│   ├── requirements.txt           # Danh sách thư viện (Windows / Linux)
│   └── requirements-mac.txt       # Danh sách thư viện (macOS)
├── OmniVoice/                     # Pipeline mô hình OmniVoice
│   ├── modelOmniLocal/            # Thư mục chứa model OmniVoice Tiếng Việt
│   ├── model_ASR_chunkformer_local# Model ASR Chunkformer VIE
│   ├── model_higgs_audio_v2_tokenizer_local # Higgs Tokenizer
│   ├── omnivoice/                 # Kiến trúc mô hình OmniVoice
│   └── omnivoice_inference/       # Logic suy luận (ttsOmni.py)
├── ui_app_Support/                # Giao diện & CSS Dark Studio Theme
└── wavs/                          # Thư mục chứa các giọng nói mẫu (.wav + .txt)
    ├── ngochuyen.wav
    ├── ngochuyen.txt
    └── ...
```

---

## 🧠 Các mô hình AI cần thiết

Khi chạy trên **Google Colab**, file notebook sẽ tự động tải các model này. Khi cài đặt **cục bộ trên máy tính**, tải 3 model sau và đặt vào đúng thư mục:

1. **Higgs Audio V2 Tokenizer** ➔ `OmniVoice/model_higgs_audio_v2_tokenizer_local`  
   👉 [huggingface.co/eustlb/higgs-audio-v2-tokenizer](https://huggingface.co/eustlb/higgs-audio-v2-tokenizer)

2. **ASR Chunkformer Tiếng Việt** ➔ `OmniVoice/model_ASR_chunkformer_local`  
   👉 [huggingface.co/khanhld/chunkformer-ctc-large-vie](https://huggingface.co/khanhld/chunkformer-ctc-large-vie)

3. **OmniVoice Tiếng Việt (KhanhTTS)** ➔ `OmniVoice/modelOmniLocal`  
   👉 [huggingface.co/kjanh/KhanhTTS-OmniVoice](https://huggingface.co/kjanh/KhanhTTS-OmniVoice)

---

## 🚀 Hướng dẫn cài đặt & sử dụng

### Cách 1: Chạy trên Google Colab (Không cần GPU máy tính)

1. Mở notebook `OmniVoice_Studio_Colab.ipynb` trực tiếp trên Google Colab.
2. Vào menu **Runtime** ➔ **Change runtime type** ➔ Chọn **T4 GPU** ➔ Bấm **Save**.
3. Bấm **Runtime** ➔ **Run all** (hoặc nhấn `Ctrl + F9`).
4. Giao diện Web UI sẽ hiển thị kèm link công khai `gradio.live`.

### Cách 2: Chạy cục bộ trên Windows

#### Yêu cầu hệ thống:
- **Python**: 3.10 hoặc 3.11
- **GPU NVIDIA**: Tối thiểu 6GB VRAM (khuyên dùng 8GB VRAM trở lên)

#### Các bước cài đặt:
```bash
# 1. Clone repository
git clone https://github.com/vietlh93/OmniVoice-Studio-Pro.git
cd OmniVoice-Studio-Pro

# 2. Tạo môi trường ảo venv
python -m venv venv

# 3. Kích hoạt venv
venv\Scripts\activate

# 4. Cài đặt thư viện dependencies
pip install -r general/requirements.txt

# 5. Khởi chạy ứng dụng
python app.py
```
*(Sau lần cài đặt đầu tiên, các lần sau chỉ cần nhấp đúp file `runApp.bat` để chạy ứng dụng).*

---

## 💡 Lưu ý về giọng mẫu (Reference Sound)

1. Để đạt chất lượng nhân bản tốt nhất, đặt file âm thanh mẫu `.wav` và file bóc băng `.txt` trùng tên trong thư mục `wavs/` (Ví dụ: `wavs/ngoc_huyen.wav` và `wavs/ngoc_huyen.txt`).
2. Thời lượng mẫu khuyến nghị từ **3 đến 10 giây**, giọng đọc rõ ràng, không lẫn nhạc nền hoặc tạp âm lớn.
3. Người dùng cũng có thể tải trực tiếp file âm thanh bất kỳ qua ô **"Upload / Record Reference Audio"** trên giao diện Web UI.

---

## Star History

<a href="https://www.star-history.com/?repos=vietlh93%2FOmniVoice-Studio-Pro&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=vietlh93/OmniVoice-Studio-Pro&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=vietlh93/OmniVoice-Studio-Pro&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=vietlh93/OmniVoice-Studio-Pro&type=date&legend=top-left" />
 </picture>
</a>
