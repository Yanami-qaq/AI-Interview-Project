import base64
import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_AUDIO_DIR = ROOT_DIR / "runtime" / "audio"
DEFAULT_SPEECH_PROVIDER = "stub"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clean_filename(value: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "_", value).strip("_") or "speech"


def save_json(audio_dir: Path, filename: str, payload: dict[str, Any]) -> str:
    try:
        ensure_dir(audio_dir)
        file_path = audio_dir / filename
        file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(file_path)
    except OSError:
        return ""


def maybe_save_audio_file(audio_dir: Path, request_id: str, payload: dict[str, Any]) -> str:
    audio_base64 = payload.get("audio_base64") or ""
    if not audio_base64:
        return ""

    audio_format = clean_filename(payload.get("audio_format") or "webm").lower()
    if audio_format.startswith("."):
        audio_format = audio_format[1:]
    try:
        file_path = audio_dir / f"{request_id}.{audio_format}"
        ensure_dir(audio_dir)
        file_path.write_bytes(base64.b64decode(audio_base64))
        return str(file_path)
    except OSError:
        return ""


def estimate_duration_ms(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", stripped))
    words = len(re.findall(r"[A-Za-z0-9_]+", stripped))
    estimated_seconds = max(1.0, chinese_chars / 4.2 + words / 2.4)
    return int(estimated_seconds * 1000)


def transcribe_audio_stub(payload: dict[str, Any], audio_dir: Path = DEFAULT_AUDIO_DIR) -> dict[str, Any]:
    request_id = uuid.uuid4().hex[:12]
    audio_dir = Path(audio_dir).resolve()
    audio_file = maybe_save_audio_file(audio_dir, request_id, payload)
    text_hint = (payload.get("text_hint") or payload.get("manual_text") or "").strip()
    transcript = text_hint
    mode = "manual_text" if transcript else "placeholder"
    warnings: list[str] = []
    if not transcript:
        warnings.append("当前为语音接口骨架，尚未接入真实 ASR 服务；请传入 text_hint/manual_text 作为转写占位结果。")

    result = {
        "request_id": request_id,
        "session_id": payload.get("session_id", ""),
        "provider": DEFAULT_SPEECH_PROVIDER,
        "mode": mode,
        "status": "completed" if transcript else "empty_transcript",
        "language": payload.get("language", "zh-CN"),
        "transcript": transcript,
        "confidence": 0.92 if transcript else 0.0,
        "duration_ms": payload.get("duration_ms") or estimate_duration_ms(transcript),
        "segments": [
            {
                "start_ms": 0,
                "end_ms": payload.get("duration_ms") or estimate_duration_ms(transcript),
                "text": transcript,
                "confidence": 0.92,
            }
        ]
        if transcript
        else [],
        "audio_file": audio_file,
        "created_at": now_iso(),
        "warnings": warnings,
        "next_action": "submit_to_interview_answer" if transcript else "retry_with_asr_or_text_hint",
    }
    save_json(audio_dir, f"{request_id}_asr.json", result)
    return result


def analyze_expression_stub(payload: dict[str, Any]) -> dict[str, Any]:
    text = (payload.get("answer_text") or payload.get("transcript") or "").strip()
    duration_ms = int(payload.get("duration_ms") or estimate_duration_ms(text) or 1)
    fillers = ["嗯", "呃", "然后然后", "就是", "那个", "这个"]
    filler_hits = [item for item in fillers if item in text]
    sentence_count = max(1, len(re.findall(r"[。！？!?；;]", text)) or 1)
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    words_per_minute = round((chinese_chars / max(duration_ms / 60000, 0.01)), 1)

    fluency_score = max(5.0, min(10.0, 9.0 - len(filler_hits) * 0.6))
    clarity_score = max(5.0, min(10.0, 6.0 + min(3.0, sentence_count * 0.8)))
    pace_score = 8.5 if 130 <= words_per_minute <= 260 else 6.5
    confidence_score = round((fluency_score + clarity_score + pace_score) / 3, 1)

    suggestions: list[str] = []
    if filler_hits:
        suggestions.append("减少口头禅和重复连接词，回答前可以先停顿 1 秒组织结构。")
    if sentence_count <= 1 and len(text) > 80:
        suggestions.append("把长句拆成 2-3 个层次，使用“首先、其次、最后”提升可听性。")
    if words_per_minute > 260:
        suggestions.append("语速偏快，建议在关键概念和结论前后加入短暂停顿。")
    if not suggestions:
        suggestions.append("表达整体较稳定，后续可继续加强示例和结论的收束感。")

    return {
        "provider": DEFAULT_SPEECH_PROVIDER,
        "status": "completed",
        "input_type": payload.get("input_type", "transcript"),
        "metrics": {
            "fluency": round(fluency_score, 1),
            "clarity": round(clarity_score, 1),
            "pace": round(pace_score, 1),
            "confidence": confidence_score,
        },
        "raw_features": {
            "duration_ms": duration_ms,
            "estimated_chars_per_minute": words_per_minute,
            "sentence_count": sentence_count,
            "filler_words": filler_hits,
        },
        "suggestions": suggestions,
        "created_at": now_iso(),
    }


def synthesize_speech_stub(payload: dict[str, Any], audio_dir: Path = DEFAULT_AUDIO_DIR) -> dict[str, Any]:
    text = (payload.get("text") or "").strip()
    if not text:
        raise ValueError("Missing text for speech synthesis.")

    utterance_id = uuid.uuid4().hex[:12]
    audio_dir = Path(audio_dir).resolve()
    result = {
        "utterance_id": utterance_id,
        "provider": DEFAULT_SPEECH_PROVIDER,
        "status": "text_only",
        "voice": payload.get("voice", "interviewer_female_zh"),
        "format": payload.get("format", "mp3"),
        "text": text,
        "audio_url": "",
        "audio_base64": "",
        "duration_ms": estimate_duration_ms(text),
        "captions": [
            {
                "start_ms": 0,
                "end_ms": estimate_duration_ms(text),
                "text": text,
            }
        ],
        "created_at": now_iso(),
        "warnings": ["当前为 TTS 预留接口，暂不生成真实音频；前端可先使用 text 字段播报或展示字幕。"],
    }
    save_json(audio_dir, f"{utterance_id}_tts.json", result)
    return result


def speech_capabilities() -> dict[str, Any]:
    return {
        "provider": DEFAULT_SPEECH_PROVIDER,
        "asr": {
            "enabled": True,
            "mode": "stub",
            "accepted_fields": ["audio_base64", "audio_format", "duration_ms", "language", "text_hint", "manual_text"],
        },
        "tts": {
            "enabled": True,
            "mode": "stub",
            "accepted_fields": ["text", "voice", "format"],
        },
        "expression_analysis": {
            "enabled": True,
            "mode": "rule_stub",
            "metrics": ["fluency", "clarity", "pace", "confidence"],
        },
    }
