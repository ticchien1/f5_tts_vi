# F5-TTS Vietnamese
Mô hình Text-to-Speech tiếng Việt dựa trên kiến trúc [F5-TTS](https://github.com/SWivid/F5-TTS).

## Nghe thử (Audio Preview)
Dưới đây là mẫu audio được tạo bởi mô hình:

<video src="previews/6935b223ac9ae405b3132fe8_1766128850313_ead83ffe.wav" controls title="Vietnamese TTS Demo"></video>

> Nếu không phát được, bạn có thể [tải file tại đây](previews/6935b223ac9ae405b3132fe8_1766128850313_ead83ffe.wav).

| Mẫu | File |
|------|------|
| Mẫu 1 | [🔊 Nghe / Tải](previews/6935b223ac9ae405b3132fe8_1766128850313_ead83ffe.wav) |

## Cài đặt
```bash
git clone https://github.com/<your-username>/F5-TTS-Vietnamese.git
cd F5-TTS-Vietnamese
pip install -e .
```

## Sử dụng

### Inference qua CLI
```bash
f5-tts_infer-cli \
  --model F5TTS_v1_Base \
  --ref_audio "ref_audio.wav" \
  --ref_text "Nội dung của audio tham chiếu." \
  --gen_text "Đoạn văn bản bạn muốn chuyển thành giọng nói."
```

### Inference qua Gradio UI
```bash
f5-tts_infer-gradio --inbrowser
```

## Cấu trúc thư mục
```
F5-TTS-Vietnamese/
├── previews/          # Các mẫu audio demo
├── src/f5_tts/
│   ├── infer/         # Code inference
│   ├── train/         # Code huấn luyện
│   ├── model/         # Kiến trúc mô hình
│   └── eval/          # Đánh giá mô hình
└── README.md
```

## Giấy phép
Vui lòng tham khảo giấy phép gốc của [F5-TTS](https://github.com/SWivid/F5-TTS).
