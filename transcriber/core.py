import subprocess
import sys
import tempfile
from pathlib import Path

from faster_whisper import WhisperModel


SUPPORTED_EXTENSIONS = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".flv", ".wmv", ".mp3", ".wav", ".flac", ".ogg", ".m4a"}


def extract_audio(video_path: Path, audio_path: Path) -> None:
    """Extract audio from video file using ffmpeg."""
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vn",                # no video
        "-acodec", "pcm_s16le",  # 16-bit PCM
        "-ar", "16000",       # 16kHz (optimal for Whisper)
        "-ac", "1",           # mono
        "-y",                 # overwrite
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error: {result.stderr}", file=sys.stderr)
        sys.exit(1)


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def transcribe(input_path: Path, model_size: str = "medium", language: str | None = None, timestamps: bool = False) -> str:
    """Transcribe a video/audio file and return markdown content."""
    if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        print(f"Unsupported format: {input_path.suffix}", file=sys.stderr)
        print(f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}", file=sys.stderr)
        sys.exit(1)

    audio_extensions = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
    is_audio = input_path.suffix.lower() in audio_extensions

    print(f"Loading model '{model_size}'...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    if is_audio:
        audio_file = str(input_path)
        tmp_dir = None
    else:
        tmp_dir = tempfile.mkdtemp()
        audio_file = str(Path(tmp_dir) / "audio.wav")
        print("Extracting audio...")
        extract_audio(input_path, Path(audio_file))

    print("Transcribing...")
    segments, info = model.transcribe(audio_file, language=language, beam_size=5)

    lines: list[str] = []
    lines.append(f"# Transcription: {input_path.name}\n")
    lines.append(f"- **Language:** {info.language} ({info.language_probability:.0%} confidence)")
    lines.append(f"- **Duration:** {format_timestamp(info.duration)}")
    lines.append(f"- **Model:** whisper-{model_size}")
    lines.append("")

    lines.append("## Content\n")

    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        if timestamps:
            ts = format_timestamp(segment.start)
            lines.append(f"**[{ts}]** {text}\n")
        else:
            lines.append(f"{text}\n")

    # Cleanup temp audio
    if tmp_dir:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return "\n".join(lines)
