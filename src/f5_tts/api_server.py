"""
F5-TTS Vietnamese REST API Server
Sử dụng FastAPI để cung cấp API cho text-to-speech
"""

import io
import os
import base64
import random
import sys
import tempfile
import json
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Audio processing
from pydub import AudioSegment
from pydub.effects import normalize as pydub_normalize
from pydub.silence import detect_leading_silence

from f5_tts.infer.utils_infer import (
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
)
from f5_tts.model import DiT, UNetT  # noqa: F401
from f5_tts.model.utils import seed_everything
from omegaconf import OmegaConf
from importlib.resources import files

# ============== Configuration ==============

CKPT_FILE = os.environ.get("F5_CKPT_FILE", "F5-TTS-Vietnamese-Test/model_870000.pt")
VOCAB_FILE = os.environ.get("F5_VOCAB_FILE", "F5-TTS-Vietnamese-Test/vocab_en_vi.txt")
MODEL_NAME = os.environ.get("F5_MODEL", "F5TTS_Base")
VOCODER_NAME = os.environ.get("F5_VOCODER", "vocos")
HOST = os.environ.get("F5_HOST", "0.0.0.0")
PORT = int(os.environ.get("F5_PORT", "8000"))
VOICES_CONFIG = os.environ.get("F5_VOICES_CONFIG", "voices.json")

# ============== Voice Configuration ==============
# Danh sách các giọng nói được cấu hình sẵn

DEFAULT_VOICES: Dict[str, dict] = {
    "nu_miennam": {
        "name": "Nữ miền Nam",
        "description": "Giọng nữ miền Nam, nhẹ nhàng tự nhiên",
        "ref_audio": "audio_nu_miennam.mp3",
        "ref_text": "Những cái lựa chọn mà bạn đưa ra á nó hoàn toàn là lựa chọn của bạn thì sẽ rất khó để bạn hoàn toàn chịu trách nhiệm cho những lựa chọn",
        "default_speed": 0.9,
        "default_cfg_strength": 3.0,
        "default_nfe_step": 64,
        "default_max_chars": 180,
        "default_pause_duration": 0.2,
    },
    # Thêm các giọng khác ở đây
    # "nam_mienbac": {
    #     "name": "Nam miền Bắc",
    #     "description": "Giọng nam miền Bắc, chuẩn mực",
    #     "ref_audio": "audio_nam_mienbac.mp3",
    #     "ref_text": "...",
    #     "default_speed": 1.0,
    #     "default_cfg_strength": 2.0,
    #     "default_nfe_step": 32,
    #     "default_max_chars": None,
    #     "default_pause_duration": 0.0,
    # },
}

# ============== Global State ==============

tts_model = None
vocoder = None
device = None
mel_spec_type = None
target_sample_rate = None

# Cached voices với processed audio
voices: Dict[str, dict] = {}

# Audio normalization constants
NORMALIZE_TARGET_SAMPLE_RATE = 24000  # F5-TTS yêu cầu 24kHz
NORMALIZE_TARGET_DB = -20.0  # Mức âm lượng chuẩn hóa (dBFS)


def trim_silence(audio, silence_thresh=-50, chunk_size=10):
    """Loại bỏ silence đầu và cuối"""
    start_trim = detect_leading_silence(audio, silence_threshold=silence_thresh, chunk_size=chunk_size)
    end_trim = detect_leading_silence(audio.reverse(), silence_threshold=silence_thresh, chunk_size=chunk_size)
    
    duration = len(audio)
    trimmed = audio[start_trim:duration - end_trim]
    
    # Đảm bảo không trim quá nhiều
    if len(trimmed) < 500:  # ít nhất 0.5 giây
        return audio
    
    return trimmed


def normalize_audio_bytes(audio_bytes: bytes, original_filename: str, 
                          target_sr: int = 24000, target_db: float = -20.0) -> tuple:
    """
    Chuẩn hóa audio từ bytes:
    - Sample rate: 24000 Hz
    - Mono channel
    - Normalize volume
    - Trim silence
    
    Returns: (normalized_audio_path, original_info, new_info)
    """
    # Lưu bytes vào temp file để pydub đọc
    ext = Path(original_filename).suffix.lower() or ".wav"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_in:
        tmp_in.write(audio_bytes)
        tmp_in_path = tmp_in.name
    
    try:
        # Load audio với pydub
        audio = AudioSegment.from_file(tmp_in_path)
        
        original_info = {
            "sample_rate": audio.frame_rate,
            "channels": audio.channels,
            "dBFS": round(audio.dBFS, 2),
            "duration": round(len(audio) / 1000, 2)
        }
        
        # 1. Chuyển về mono
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # 2. Chuyển về target sample rate
        if audio.frame_rate != target_sr:
            audio = audio.set_frame_rate(target_sr)
        
        # 3. Trim silence đầu/cuối
        audio = trim_silence(audio)
        
        # 4. Normalize volume
        audio = pydub_normalize(audio)
        change_in_dBFS = target_db - audio.dBFS
        audio = audio.apply_gain(change_in_dBFS)
        
        # 5. Export to temp wav file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
            audio.export(tmp_out.name, format="wav")
            output_path = tmp_out.name
        
        new_info = {
            "sample_rate": target_sr,
            "channels": 1,
            "dBFS": round(audio.dBFS, 2),
            "duration": round(len(audio) / 1000, 2)
        }
        
        return output_path, original_info, new_info
        
    finally:
        # Cleanup input temp file
        if os.path.exists(tmp_in_path):
            os.unlink(tmp_in_path)


def load_voices_config():
    """Load cấu hình voices từ file JSON nếu có"""
    global DEFAULT_VOICES
    
    if os.path.exists(VOICES_CONFIG):
        try:
            with open(VOICES_CONFIG, 'r', encoding='utf-8') as f:
                loaded_voices = json.load(f)
                DEFAULT_VOICES.update(loaded_voices)
                print(f"📁 Loaded {len(loaded_voices)} voices from {VOICES_CONFIG}")
        except Exception as e:
            print(f"⚠️ Không thể load {VOICES_CONFIG}: {e}")


def save_voices_config():
    """Lưu cấu hình voices ra file JSON"""
    try:
        # Tạo bản sao không chứa cached data
        voices_to_save = {}
        for voice_id, voice_data in voices.items():
            voices_to_save[voice_id] = {
                "name": voice_data.get("name", ""),
                "description": voice_data.get("description", ""),
                "ref_audio": voice_data.get("ref_audio_path", voice_data.get("ref_audio", "")),
                "ref_text": voice_data.get("original_ref_text", voice_data.get("ref_text", "")),
                "default_speed": voice_data.get("default_speed", 1.0),
                "default_cfg_strength": voice_data.get("default_cfg_strength", 2.0),
            }
        
        with open(VOICES_CONFIG, 'w', encoding='utf-8') as f:
            json.dump(voices_to_save, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved voices config to {VOICES_CONFIG}")
    except Exception as e:
        print(f"⚠️ Không thể lưu {VOICES_CONFIG}: {e}")


def load_tts_model():
    """Load TTS model và vocoder một lần khi khởi động"""
    global tts_model, vocoder, device, mel_spec_type, target_sample_rate
    global voices

    import torch

    device = (
        "cuda" if torch.cuda.is_available()
        else "xpu" if torch.xpu.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"🔧 Loading model on device: {device}")
    print(f"📁 Checkpoint: {CKPT_FILE}")
    print(f"📁 Vocab: {VOCAB_FILE}")

    # Load config
    model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{MODEL_NAME}.yaml")))
    model_cls = globals()[model_cfg.model.backbone]
    model_arc = model_cfg.model.arch

    mel_spec_type = VOCODER_NAME
    target_sample_rate = model_cfg.model.mel_spec.target_sample_rate

    # Load vocoder
    print(f"🔊 Loading vocoder: {mel_spec_type}")
    vocoder = load_vocoder(mel_spec_type, False, None, device)

    # Load TTS model
    print("🤖 Loading TTS model...")
    tts_model = load_model(
        model_cls, model_arc, CKPT_FILE, mel_spec_type, VOCAB_FILE, "euler", True, device
    )

    # Load voices config
    load_voices_config()

    # Pre-process all voices
    print("🎤 Pre-processing voices...")
    for voice_id, voice_config in DEFAULT_VOICES.items():
        ref_audio_path = voice_config.get("ref_audio", "")
        ref_text = voice_config.get("ref_text", "")
        
        if os.path.exists(ref_audio_path):
            try:
                processed_audio, processed_text = preprocess_ref_audio_text(
                    ref_audio_path, ref_text, device=device
                )
                voices[voice_id] = {
                    **voice_config,
                    "ref_audio_path": ref_audio_path,
                    "original_ref_text": ref_text,
                    "processed_audio": processed_audio,
                    "processed_text": processed_text,
                }
                print(f"  ✅ {voice_id}: {voice_config.get('name', 'Unknown')}")
            except Exception as e:
                print(f"  ❌ {voice_id}: Lỗi - {e}")
        else:
            print(f"  ⚠️ {voice_id}: File không tồn tại - {ref_audio_path}")

    print(f"✅ Model loaded! {len(voices)} voices available.")


# ============== API Models ==============

class VoiceInfo(BaseModel):
    """Thông tin một giọng nói"""
    id: str
    name: str
    description: str
    preview_url: Optional[str] = None  # URL để nghe thử giọng nói
    default_speed: float = 1.0
    default_cfg_strength: float = 2.0
    default_nfe_step: int = 32
    default_max_chars: Optional[int] = None
    default_pause_duration: float = 0.0


class TTSConfig(BaseModel):
    """Cấu hình cho TTS"""
    speed: float = 1.0
    nfe_step: int = 32
    cfg_strength: float = 2.0
    cross_fade_duration: float = 0.15
    pause_duration: float = 0.0  # Khoảng dừng giữa các đoạn (giây)
    max_chars: Optional[int] = None  # Số ký tự tối đa mỗi chunk, None = auto
    remove_silence: bool = False
    seed: Optional[int] = None


class TTSRequest(BaseModel):
    """Request model cho TTS inference"""
    voice_id: str
    gen_text: str
    config: Optional[TTSConfig] = None
    output_format: str = "wav"  # wav hoặc base64


class TTSResponse(BaseModel):
    """Response model cho TTS inference (khi output_format=base64)"""
    success: bool
    message: str
    audio_base64: Optional[str] = None
    sample_rate: Optional[int] = None
    seed: Optional[int] = None
    duration: Optional[float] = None


class AddVoiceRequest(BaseModel):
    """Request để thêm voice mới"""
    voice_id: str
    name: str
    description: str = ""
    ref_text: str
    default_speed: float = 1.0
    default_cfg_strength: float = 2.0


# ============== FastAPI App ==============

app = FastAPI(
    title="F5-TTS Vietnamese API",
    description="REST API cho F5-TTS Vietnamese Text-to-Speech",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler for better error messages
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print("=" * 60)
    print("❌ VALIDATION ERROR")
    print("=" * 60)
    print(f"URL: {request.url}")
    print(f"Method: {request.method}")
    print(f"Headers: {dict(request.headers)}")
    print(f"Errors: {exc.errors()}")
    print("=" * 60)
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "message": "Validation error - kiểm tra dữ liệu gửi lên",
            "hint": "Đảm bảo gửi multipart/form-data với các field: ref_audio (file), ref_text (text), gen_text (text)"
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    import traceback
    print("=" * 60)
    print("❌ UNHANDLED EXCEPTION")
    print("=" * 60)
    print(f"URL: {request.url}")
    print(f"Method: {request.method}")
    print(f"Error type: {type(exc).__name__}")
    print(f"Error message: {str(exc)}")
    traceback.print_exc()
    print("=" * 60)
    return JSONResponse(
        status_code=500,
        content={
            "detail": str(exc),
            "error_type": type(exc).__name__,
            "message": "Internal server error"
        }
    )


@app.on_event("startup")
async def startup_event():
    """Load model khi server khởi động"""
    load_tts_model()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "F5-TTS Vietnamese API is running",
        "device": device,
        "model": MODEL_NAME,
        "voices_count": len(voices)
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": tts_model is not None}


# ============== Voice Management ==============

@app.get("/voices", response_model=List[VoiceInfo])
async def get_voices():
    """
    Lấy danh sách tất cả giọng nói có sẵn
    """
    voice_list = []
    for voice_id, voice_data in voices.items():
        # Tạo preview URL - luôn có vì sẽ fallback về ref_audio
        preview_url = f"/voices/{voice_id}/preview"
        
        voice_list.append(VoiceInfo(
            id=voice_id,
            name=voice_data.get("name", voice_id),
            description=voice_data.get("description", ""),
            preview_url=preview_url,
            default_speed=voice_data.get("default_speed", 1.0),
            default_cfg_strength=voice_data.get("default_cfg_strength", 2.0),
            default_nfe_step=voice_data.get("default_nfe_step", 32),
            default_max_chars=voice_data.get("default_max_chars"),
            default_pause_duration=voice_data.get("default_pause_duration", 0.0),
        ))
    return voice_list


@app.get("/voices/{voice_id}", response_model=VoiceInfo)
async def get_voice(voice_id: str):
    """
    Lấy thông tin chi tiết một giọng nói
    """
    if voice_id not in voices:
        raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' không tồn tại")
    
    voice_data = voices[voice_id]
    preview_url = f"/voices/{voice_id}/preview"
    
    return VoiceInfo(
        id=voice_id,
        name=voice_data.get("name", voice_id),
        description=voice_data.get("description", ""),
        preview_url=preview_url,
        default_speed=voice_data.get("default_speed", 1.0),
        default_cfg_strength=voice_data.get("default_cfg_strength", 2.0),
        default_nfe_step=voice_data.get("default_nfe_step", 32),
        default_max_chars=voice_data.get("default_max_chars"),
        default_pause_duration=voice_data.get("default_pause_duration", 0.0),
    )


@app.get("/voices/{voice_id}/preview")
async def get_voice_preview(voice_id: str):
    """
    Lấy audio preview của một giọng nói
    
    Trả về file audio để nghe thử giọng
    """
    if voice_id not in voices:
        raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' không tồn tại")
    
    voice_data = voices[voice_id]
    
    # Ưu tiên preview_audio, fallback về ref_audio nếu không có
    preview_path = voice_data.get("preview_audio")
    if not preview_path or not os.path.exists(preview_path):
        preview_path = voice_data.get("ref_audio_path") or voice_data.get("ref_audio")
    
    if not preview_path or not os.path.exists(preview_path):
        raise HTTPException(status_code=404, detail=f"Preview audio không tồn tại cho voice '{voice_id}'")
    
    # Xác định content type dựa trên extension
    ext = Path(preview_path).suffix.lower()
    content_types = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
    }
    content_type = content_types.get(ext, "audio/mpeg")
    
    def iter_file():
        with open(preview_path, "rb") as f:
            yield from f
    
    return StreamingResponse(
        iter_file(),
        media_type=content_type,
        headers={
            "Content-Disposition": f"inline; filename={voice_id}_preview{ext}",
            "Accept-Ranges": "bytes",
        }
    )


@app.post("/voices/add")
async def add_voice(
    voice_id: str = Form(...),
    name: str = Form(...),
    description: str = Form(""),
    ref_text: str = Form(...),
    ref_audio: UploadFile = File(...),
    default_speed: float = Form(1.0),
    default_cfg_strength: float = Form(2.0),
):
    """
    Thêm giọng nói mới
    
    Upload file audio tham chiếu và cấu hình giọng nói
    """
    if voice_id in voices:
        raise HTTPException(status_code=400, detail=f"Voice '{voice_id}' đã tồn tại")

    try:
        # Tạo folder lưu voice audio
        voices_dir = Path("voices")
        voices_dir.mkdir(exist_ok=True)
        
        # Lưu file audio
        file_ext = Path(ref_audio.filename).suffix
        audio_path = voices_dir / f"{voice_id}{file_ext}"
        
        content = await ref_audio.read()
        with open(audio_path, "wb") as f:
            f.write(content)

        # Process audio
        processed_audio, processed_text = preprocess_ref_audio_text(
            str(audio_path), ref_text, device=device
        )

        # Lưu vào voices dict
        voices[voice_id] = {
            "name": name,
            "description": description,
            "ref_audio": str(audio_path),
            "ref_audio_path": str(audio_path),
            "ref_text": ref_text,
            "original_ref_text": ref_text,
            "default_speed": default_speed,
            "default_cfg_strength": default_cfg_strength,
            "processed_audio": processed_audio,
            "processed_text": processed_text,
        }

        # Lưu config
        save_voices_config()

        return {
            "success": True,
            "message": f"Đã thêm voice '{voice_id}'",
            "voice": VoiceInfo(
                id=voice_id,
                name=name,
                description=description,
                default_speed=default_speed,
                default_cfg_strength=default_cfg_strength,
            )
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi thêm voice: {str(e)}")


@app.delete("/voices/{voice_id}")
async def delete_voice(voice_id: str):
    """
    Xóa một giọng nói
    """
    if voice_id not in voices:
        raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' không tồn tại")

    del voices[voice_id]
    save_voices_config()

    return {"success": True, "message": f"Đã xóa voice '{voice_id}'"}


# ============== TTS Inference ==============

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """
    Chuyển đổi văn bản thành giọng nói
    
    - **voice_id**: ID của giọng nói (lấy từ /voices)
    - **gen_text**: Văn bản cần chuyển đổi
    - **config**: Cấu hình TTS (tùy chọn)
    - **output_format**: "wav" (trả file) hoặc "base64" (trả JSON)
    """
    if tts_model is None:
        raise HTTPException(status_code=503, detail="Model chưa được load")

    if request.voice_id not in voices:
        raise HTTPException(
            status_code=404, 
            detail=f"Voice '{request.voice_id}' không tồn tại. Sử dụng GET /voices để xem danh sách."
        )

    if not request.gen_text.strip():
        raise HTTPException(status_code=400, detail="gen_text không được để trống")

    try:
        voice = voices[request.voice_id]
        config = request.config or TTSConfig()

        # Use voice defaults if not specified in config
        speed = config.speed if config.speed != 1.0 else voice.get("default_speed", 1.0)
        cfg_strength = config.cfg_strength if config.cfg_strength != 2.0 else voice.get("default_cfg_strength", 2.0)
        nfe_step = config.nfe_step if config.nfe_step != 32 else voice.get("default_nfe_step", 32)
        max_chars = config.max_chars if config.max_chars is not None else voice.get("default_max_chars")
        pause_duration = config.pause_duration if config.pause_duration != 0.0 else voice.get("default_pause_duration", 0.0)

        # Set seed
        seed = config.seed if config.seed is not None else random.randint(0, sys.maxsize)
        seed_everything(seed)

        # Run inference
        wav, sr, _ = infer_process(
            voice["processed_audio"],
            voice["processed_text"],
            request.gen_text,
            tts_model,
            vocoder,
            mel_spec_type,
            show_info=print,
            progress=None,
            target_rms=0.1,
            cross_fade_duration=config.cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=-1,
            speed=speed,
            fix_duration=None,
            device=device,
            max_chars=max_chars,
            pause_duration=pause_duration,
        )

        # Handle silence removal
        if config.remove_silence:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, wav, sr)
                remove_silence_for_generated_wav(tmp.name)
                wav, sr = sf.read(tmp.name)
                os.unlink(tmp.name)

        # Calculate duration
        duration = len(wav) / sr

        # Convert to bytes
        buffer = io.BytesIO()
        sf.write(buffer, wav, sr, format='WAV')
        buffer.seek(0)

        if request.output_format == "base64":
            audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            return TTSResponse(
                success=True,
                message="Tạo audio thành công",
                audio_base64=audio_base64,
                sample_rate=sr,
                seed=seed,
                duration=round(duration, 2)
            )
        else:
            return StreamingResponse(
                buffer,
                media_type="audio/wav",
                headers={
                    "Content-Disposition": f"attachment; filename=tts_{request.voice_id}.wav",
                    "X-Seed": str(seed),
                    "X-Sample-Rate": str(sr),
                    "X-Duration": str(round(duration, 2))
                }
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi inference: {str(e)}")


@app.post("/tts/quick")
async def quick_tts(
    voice_id: str = Form(...),
    gen_text: str = Form(...),
    speed: float = Form(1.0),
    nfe_step: int = Form(32),
    cfg_strength: float = Form(2.0),
    max_chars: Optional[int] = Form(None),
    pause_duration: float = Form(0.0),
    remove_silence: bool = Form(False),
):
    """
    TTS nhanh với form data (tiện cho test từ HTML form)
    
    Trả về file WAV trực tiếp
    """
    request = TTSRequest(
        voice_id=voice_id,
        gen_text=gen_text,
        config=TTSConfig(
            speed=speed,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            max_chars=max_chars,
            pause_duration=pause_duration,
            remove_silence=remove_silence,
        ),
        output_format="wav"
    )
    return await text_to_speech(request)


# ============== Voice Clone ==============

@app.post("/voice-clone/debug")
async def voice_clone_debug(request: Request):
    """
    Debug endpoint để kiểm tra request data
    """
    print("=" * 60)
    print("🔍 DEBUG: Voice Clone Request")
    print("=" * 60)
    
    # Log headers
    print("Headers:")
    for key, value in request.headers.items():
        print(f"  {key}: {value}")
    
    # Check content type
    content_type = request.headers.get("content-type", "")
    print(f"\nContent-Type: {content_type}")
    
    if "multipart/form-data" not in content_type:
        print("⚠️ WARNING: Content-Type không phải multipart/form-data!")
        return JSONResponse(
            status_code=400,
            content={
                "error": "Content-Type phải là multipart/form-data",
                "received": content_type,
                "hint": "Sử dụng FormData trong JavaScript hoặc -F trong curl"
            }
        )
    
    try:
        # Try to parse form data
        form = await request.form()
        print(f"\nForm fields received: {list(form.keys())}")
        
        result = {"fields": {}}
        for key in form.keys():
            value = form[key]
            if hasattr(value, 'filename'):  # It's a file
                content = await value.read()
                result["fields"][key] = {
                    "type": "file",
                    "filename": value.filename,
                    "content_type": value.content_type,
                    "size": len(content)
                }
                print(f"  {key}: FILE - {value.filename} ({len(content)} bytes)")
            else:
                result["fields"][key] = {
                    "type": "text",
                    "value": str(value)[:100] + "..." if len(str(value)) > 100 else str(value),
                    "length": len(str(value))
                }
                print(f"  {key}: TEXT - {str(value)[:50]}...")
        
        print("=" * 60)
        return {
            "status": "ok",
            "message": "Request parsed successfully",
            **result
        }
        
    except Exception as e:
        import traceback
        print(f"❌ Error parsing form: {e}")
        traceback.print_exc()
        print("=" * 60)
        return JSONResponse(
            status_code=400,
            content={
                "error": str(e),
                "error_type": type(e).__name__,
                "message": "Không thể parse form data"
            }
        )


class VoiceCloneRequest(BaseModel):
    """Response model cho voice clone"""
    success: bool
    message: str
    audio_base64: Optional[str] = None
    sample_rate: Optional[int] = None
    seed: Optional[int] = None
    duration: Optional[float] = None
    audio_normalized_info: Optional[dict] = None


@app.post("/voice-clone")
async def voice_clone(
    ref_audio: UploadFile = File(..., description="File audio tham chiếu (wav, mp3, flac, ogg...)"),
    ref_text: str = Form(..., description="Văn bản tương ứng với audio tham chiếu"),
    gen_text: str = Form(..., description="Văn bản cần tổng hợp giọng nói"),
    speed: float = Form(1.0, description="Tốc độ đọc (0.5 - 2.0)"),
    nfe_step: int = Form(32, description="Số bước denoising (8-128, cao = chất lượng tốt hơn)"),
    cfg_strength: float = Form(2.0, description="Độ mạnh guidance (1.0-5.0)"),
    cross_fade_duration: float = Form(0.15, description="Thời gian cross-fade giữa các đoạn"),
    max_chars: Optional[int] = Form(None, description="Số ký tự tối đa mỗi chunk"),
    pause_duration: float = Form(0.0, description="Khoảng dừng giữa các đoạn (giây)"),
    remove_silence: bool = Form(False, description="Loại bỏ khoảng lặng dài"),
    normalize_audio: bool = Form(True, description="Chuẩn hóa audio (24kHz, mono, normalize volume)"),
    output_format: str = Form("wav", description="Format output: 'wav' hoặc 'base64'"),
    seed: Optional[int] = Form(None, description="Seed cho reproducibility"),
):
    """
    Voice Clone - Clone giọng nói từ audio tham chiếu
    
    Upload file audio và ref_text, API sẽ:
    1. Tự động chuẩn hóa audio (24kHz, mono, normalize volume, trim silence)
    2. Clone giọng nói để tổng hợp gen_text
    3. Trả về audio đã clone
    
    **Lưu ý:**
    - Audio tham chiếu nên có độ dài 5-15 giây
    - ref_text phải khớp chính xác với nội dung trong audio
    - Chất lượng audio tham chiếu càng tốt thì kết quả càng tốt
    """
    # Logging request info
    print("=" * 60)
    print("🎤 VOICE CLONE REQUEST")
    print("=" * 60)
    print(f"📁 ref_audio filename: {ref_audio.filename}")
    print(f"📁 ref_audio content_type: {ref_audio.content_type}")
    print(f"📝 ref_text: {ref_text[:100]}..." if len(ref_text) > 100 else f"📝 ref_text: {ref_text}")
    print(f"📝 gen_text: {gen_text[:100]}..." if len(gen_text) > 100 else f"📝 gen_text: {gen_text}")
    print(f"⚙️ speed={speed}, nfe_step={nfe_step}, cfg_strength={cfg_strength}")
    print(f"⚙️ normalize_audio={normalize_audio}, output_format={output_format}")
    print("-" * 60)
    
    if tts_model is None:
        print("❌ ERROR: Model chưa được load")
        raise HTTPException(status_code=503, detail="Model chưa được load")
    
    if not ref_text.strip():
        raise HTTPException(status_code=400, detail="ref_text không được để trống")
    
    if not gen_text.strip():
        raise HTTPException(status_code=400, detail="gen_text không được để trống")
    
    normalized_audio_path = None
    
    try:
        # Đọc audio bytes
        print("📥 Reading audio bytes...")
        audio_bytes = await ref_audio.read()
        print(f"📥 Audio size: {len(audio_bytes)} bytes")
        
        if len(audio_bytes) == 0:
            print("❌ ERROR: Audio file rỗng!")
            raise HTTPException(status_code=400, detail="Audio file rỗng")
        
        audio_info = None
        
        if normalize_audio:
            # Chuẩn hóa audio
            print("🔧 Normalizing audio...")
            normalized_audio_path, original_info, new_info = normalize_audio_bytes(
                audio_bytes,
                ref_audio.filename or "audio.wav",
                target_sr=NORMALIZE_TARGET_SAMPLE_RATE,
                target_db=NORMALIZE_TARGET_DB
            )
            audio_path_for_processing = normalized_audio_path
            audio_info = {
                "original": original_info,
                "normalized": new_info
            }
            print(f"✅ Audio normalized: {original_info} -> {new_info}")
        else:
            # Lưu audio gốc vào temp file
            ext = Path(ref_audio.filename).suffix.lower() or ".wav"
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(audio_bytes)
                audio_path_for_processing = tmp.name
                normalized_audio_path = tmp.name
        
        # Pre-process audio và text
        print("🔄 Pre-processing audio and text...")
        processed_audio, processed_text = preprocess_ref_audio_text(
            audio_path_for_processing, ref_text, device=device
        )
        print(f"✅ Pre-processing done. Processed text: {processed_text[:50]}...")
        
        # Set seed
        actual_seed = seed if seed is not None else random.randint(0, sys.maxsize)
        seed_everything(actual_seed)
        print(f"🎲 Using seed: {actual_seed}")
        
        # Run inference
        print("🚀 Running inference...")
        wav, sr, _ = infer_process(
            processed_audio,
            processed_text,
            gen_text,
            tts_model,
            vocoder,
            mel_spec_type,
            show_info=print,
            progress=None,
            target_rms=0.1,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=-1,
            speed=speed,
            fix_duration=None,
            device=device,
            max_chars=max_chars,
            pause_duration=pause_duration,
        )
        
        # Handle silence removal
        if remove_silence:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, wav, sr)
                remove_silence_for_generated_wav(tmp.name)
                wav, sr = sf.read(tmp.name)
                os.unlink(tmp.name)
        
        # Calculate duration
        duration = len(wav) / sr
        print(f"✅ Inference done! Duration: {duration:.2f}s, Sample rate: {sr}")
        
        # Convert to bytes
        buffer = io.BytesIO()
        sf.write(buffer, wav, sr, format='WAV')
        buffer.seek(0)
        
        print(f"📤 Sending response (format: {output_format})")
        print("=" * 60)
        
        if output_format == "base64":
            audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            return VoiceCloneRequest(
                success=True,
                message="Voice clone thành công",
                audio_base64=audio_base64,
                sample_rate=sr,
                seed=actual_seed,
                duration=round(duration, 2),
                audio_normalized_info=audio_info
            )
        else:
            return StreamingResponse(
                buffer,
                media_type="audio/wav",
                headers={
                    "Content-Disposition": "attachment; filename=voice_clone_output.wav",
                    "X-Seed": str(actual_seed),
                    "X-Sample-Rate": str(sr),
                    "X-Duration": str(round(duration, 2))
                }
            )
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print("=" * 60)
        print("❌ VOICE CLONE ERROR")
        print("=" * 60)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        print("=" * 60)
        raise HTTPException(status_code=500, detail=f"Lỗi voice clone: {str(e)}")
    
    finally:
        # Cleanup temp files
        if normalized_audio_path and os.path.exists(normalized_audio_path):
            try:
                os.unlink(normalized_audio_path)
            except:
                pass


# ============== Config Defaults ==============

@app.get("/config/defaults")
async def get_default_config():
    """
    Lấy cấu hình mặc định cho TTS
    """
    return {
        "speed": {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1, "description": "Tốc độ đọc"},
        "nfe_step": {"default": 32, "min": 8, "max": 128, "step": 8, "description": "Số bước denoising (cao = chất lượng tốt hơn, chậm hơn)"},
        "cfg_strength": {"default": 2.0, "min": 1.0, "max": 5.0, "step": 0.5, "description": "Độ mạnh classifier-free guidance"},
        "cross_fade_duration": {"default": 0.15, "min": 0.0, "max": 0.5, "step": 0.05, "description": "Thời gian cross-fade giữa các đoạn (giây)"},
        "pause_duration": {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1, "description": "Khoảng dừng giữa các đoạn (giây)"},
        "max_chars": {"default": None, "min": 50, "max": 300, "step": 10, "description": "Số ký tự tối đa mỗi chunk (None = tự động)"},
        "remove_silence": {"default": False, "description": "Loại bỏ khoảng lặng dài"},
    }


def main():
    """Entry point để chạy server"""
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           F5-TTS Vietnamese API Server                       ║
╠══════════════════════════════════════════════════════════════╣
║  Endpoints:                                                  ║
║    GET  /              - Health check                        ║
║    GET  /voices        - Danh sách giọng nói                 ║
║    GET  /voices/{{id}}   - Chi tiết giọng nói                 ║
║    POST /voices/add    - Thêm giọng mới                      ║
║    DELETE /voices/{{id}} - Xóa giọng                          ║
║    POST /tts           - Text to speech (JSON)               ║
║    POST /tts/quick     - Text to speech (Form)               ║
║    POST /voice-clone   - Voice clone từ audio upload         ║
║    GET  /config/defaults - Cấu hình mặc định                 ║
║                                                              ║
║  Docs: http://{HOST}:{PORT}/docs                               ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "f5_tts.api_server:app",
        host=HOST,
        port=PORT,
        reload=False,
        workers=1
    )


if __name__ == "__main__":
    main()
