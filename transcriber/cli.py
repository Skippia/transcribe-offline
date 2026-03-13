import argparse
import sys
from pathlib import Path

from transcriber.core import transcribe


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe video/audio files to markdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  transcribe video.mp4\n  transcribe lecture.mp4 -m medium -l en\n  transcribe podcast.mp3 -o notes.md --no-timestamps",
    )
    parser.add_argument("input", type=Path, help="Path to video or audio file")
    parser.add_argument("-o", "--output", type=Path, help="Output markdown file (default: <input>.md)")
    parser.add_argument("-m", "--model", default="medium", choices=["tiny", "base", "small", "medium", "large-v3"],
                        help="Whisper model size (default: medium)")
    parser.add_argument("-l", "--language", default=None, help="Language code, e.g. 'en', 'uk', 'ru' (default: auto-detect)")
    parser.add_argument("--timestamps", action="store_true", help="Include timestamps in output (off by default)")

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or args.input.with_suffix(".md")

    result = transcribe(
        input_path=args.input.resolve(),
        model_size=args.model,
        language=args.language,
        timestamps=args.timestamps,
    )

    output_path.write_text(result, encoding="utf-8")
    print(f"\nDone! Transcription saved to: {output_path}")


if __name__ == "__main__":
    main()
