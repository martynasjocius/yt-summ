#!/usr/bin/env python3
#
# yt-summ - YouTube Video Summarizer
#
# Downloads YouTube video content, transcribes audio, and generates a concise summary
# using LLM with different length options (short/medium/long).
#
# Copyright (c) 2025 Martynas Jocius
#

import argparse
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path


def _running_in_virtualenv() -> bool:
    base_prefix = getattr(sys, "base_prefix", sys.prefix)
    return (
        sys.prefix != base_prefix
        or hasattr(sys, "real_prefix")
        or bool(os.environ.get("VIRTUAL_ENV"))
    )


def _find_downloaded_audio(directory: Path) -> Path | None:
    """Return the first audio file found in the download directory."""
    for pattern in ("*.wav", "*.m4a", "*.mp3", "*.webm"):
        matches = sorted(directory.glob(pattern))
        if matches:
            return matches[0]
    return None


try:
    import yt_dlp
    import whisper
    import anthropic
    import openai
except ImportError as e:
    print(f"Dependency import failed: {e}")
    if not _running_in_virtualenv():
        print(
            "Activate your project virtual environment (e.g. `source venv/bin/activate`) and retry; "
            "see README for setup details."
        )
    else:
        print(
            "Install missing packages with `pip install -r requirements.txt`. See README for details."
        )
    sys.exit(1)


class YtSumm:
    def __init__(self, summary_length="short"):
        self.anthropic_client = None
        self.openai_client = None
        self.whisper_model = None
        self.summary_length = summary_length
        self._setup_llm_clients()
        self._setup_whisper()

    def _setup_llm_clients(self):
        """Initialize LLM clients based on available API keys"""
        # Check for Anthropic API key first
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            try:
                self.anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
                print("Using Anthropic Claude API")
                return
            except Exception as e:
                print(f"Failed to initialize Anthropic client: {e}")

        # Fallback to OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                self.openai_client = openai.OpenAI(api_key=openai_key)
                print("Using OpenAI API")
                return
            except Exception as e:
                print(f"Failed to initialize OpenAI client: {e}")

        print(
            "No valid API keys found. Please set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable."
        )
        sys.exit(1)

    def _setup_whisper(self):
        """Initialize Whisper model for transcription"""
        try:
            # Use base model for faster processing
            self.whisper_model = whisper.load_model("base")
        except Exception as e:
            print(f"Failed to load Whisper model: {e}")
            sys.exit(1)

    def download_video_audio(self, url: str) -> tuple[str, str]:
        """Download video and extract audio"""
        print(f"Downloading video from: {url}")
        temp_dir = Path(tempfile.mkdtemp())
        common_opts = {
            "format": "bestaudio/best",
            "outtmpl": str(temp_dir / "%(title)s.%(ext)s"),
            "extractaudio": True,
            "audioformat": "wav",
            "noplaylist": True,
            "extractor_retries": 3,
            "fragment_retries": 3,
        }
        browsers = ["chromium", "chrome", "firefox", "brave"]
        attempts = [
            (browser, {**common_opts, "cookiesfrombrowser": (browser,)})
            for browser in browsers
        ]
        attempts.append((None, common_opts))  # Final attempt without cookies

        video_title = "Unknown"
        last_error: Exception | None = None

        try:
            for browser, options in attempts:
                label = f"{browser} cookies" if browser else "no browser cookies"
                print(f"Trying download with {label}...")

                try:
                    with yt_dlp.YoutubeDL(options) as ydl:
                        info = ydl.extract_info(url, download=True)
                    video_title = info.get("title", "Unknown")

                    audio_file = _find_downloaded_audio(temp_dir)
                    if not audio_file:
                        raise RuntimeError("No audio file found after download")

                    print(f"Video title: {video_title}")
                    print(f"Audio file: {audio_file}")
                    return str(audio_file), video_title

                except Exception as error:
                    audio_file = _find_downloaded_audio(temp_dir)
                    if audio_file:
                        inferred_title = video_title or audio_file.stem
                        print(
                            f"Download completed with warnings. Audio file: {audio_file}"
                        )
                        return str(audio_file), inferred_title

                    print(f"Attempt with {label} failed: {error}")
                    last_error = error

            if last_error is not None:
                raise last_error
            raise RuntimeError("Unable to download audio track")

        except Exception as fatal_error:
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"Error downloading video: {fatal_error}")
            print("\nTroubleshooting tips:")
            print("1. Make sure you're logged into YouTube in your browser")
            print("2. Try accessing the video directly in your browser first")
            print(
                "3. Some videos may be region-restricted or require special permissions"
            )
            print("4. Try using a different browser or clearing browser data")
            sys.exit(1)

    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio to text using Whisper"""
        print("Transcribing audio...")
        try:
            result = self.whisper_model.transcribe(audio_path)
            transcript = result["text"].strip()
            print(f"Transcription completed ({len(transcript)} characters)")
            return transcript
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            sys.exit(1)

    def generate_summary(self, transcript: str) -> str:
        """Generate summary using LLM"""
        print("Generating summary...")

        # Define prompts based on length
        length_prompts = {
            "short": "Summarize the following video transcript concisely. Start with a brief topic/card name (2-4 words) in UPPERCASE that captures the main subject. Then use bullet points or numbered lists for the content. Focus on the most important key points, main ideas, and actionable insights. Keep it brief but comprehensive.",
            "medium": "Summarize the following video transcript into a medium-length format. Start with a brief topic/card name (2-4 words) in UPPERCASE that captures the main subject. Then use bullet points or numbered lists for organization. Include key points, main ideas, important details, and actionable insights. Aim for about 2-3 paragraphs or a well-structured list.",
            "long": "Provide a comprehensive summary of the following video transcript. Start with a brief topic/card name (2-4 words) in UPPERCASE that captures the main subject. Then use clear headings, bullet points, and numbered lists for organization. Include all major points, supporting details, examples, and actionable insights. This should be a thorough overview of the content.",
        }

        prompt = f"""{length_prompts[self.summary_length]}

Transcript:
{transcript}

Summary:"""

        try:
            if self.anthropic_client:
                response = self.anthropic_client.messages.create(
                    model="claude-3-haiku-20240307",  # Fast and cost-effective
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}],
                )
                # Extract token usage
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                total_tokens = (
                    response.usage.input_tokens + response.usage.output_tokens
                )
                print(
                    f"Token usage: {input_tokens} input + {output_tokens} output = {total_tokens} total"
                )
                return response.content[0].text.strip()

            elif self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=0.7,
                )
                # Extract token usage
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
                print(
                    f"Token usage: {input_tokens} input + {output_tokens} output = {total_tokens} total"
                )
                return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error generating summary: {e}")
            sys.exit(1)

        print("No language model client is configured. Check your API key setup.")
        sys.exit(1)

    def cleanup_temp_files(self, audio_path: str):
        """Clean up temporary files"""
        try:
            temp_dir = os.path.dirname(audio_path)
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass  # Ignore cleanup errors

    def extract_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL"""
        # Handle both youtube.com and youtu.be formats
        patterns = [
            r"(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})",
            r"youtube\.com/embed/([a-zA-Z0-9_-]{11})",
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        # Fallback: use a timestamp-based ID
        return f"video_{int(time.time())}"

    def generate_filename_with_llm(self, video_title: str) -> str:
        """Generate a short, descriptive filename using LLM"""
        try:
            prompt = f"""Create a short, descriptive filename prefix (2-4 words) for this YouTube video title. 
The filename should be:
- Concise and clear
- Use hyphens between words
- Lowercase only
- No special characters except hyphens
- Capture the main topic/theme

Video title: "{video_title}"

Respond with ONLY the filename prefix (no file extension, no video ID). Example: "hiking-essentials" or "python-tutorial" or "cooking-basics"."""

            if self.anthropic_client:
                response = self.anthropic_client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=50,
                    messages=[{"role": "user", "content": prompt}],
                )
                filename_prefix = response.content[0].text.strip()
            elif self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    max_tokens=50,
                    messages=[{"role": "user", "content": prompt}],
                )
                filename_prefix = response.choices[0].message.content.strip()
            else:
                return None

            # Clean up the response - remove any extra text or formatting
            filename_prefix = filename_prefix.splitlines()[0].strip().strip("\"' ")
            filename_prefix = filename_prefix.lower()
            filename_prefix = re.sub(r"[^a-z0-9-]", "-", filename_prefix)
            filename_prefix = re.sub(r"-+", "-", filename_prefix).strip("-")

            if filename_prefix:
                return filename_prefix
            return None

        except Exception as e:
            print(f"Warning: LLM filename generation failed: {e}")
            return None

    def create_descriptive_filename(self, video_title: str, video_id: str) -> str:
        """Create a descriptive filename from video title and ID"""
        # Try LLM-based generation first
        llm_prefix = self.generate_filename_with_llm(video_title)
        if llm_prefix:
            return f"{llm_prefix}-{video_id}.txt"

        # Simple fallback: use first few words from title
        clean_title = re.sub(r"[^\w\s-]", "", video_title.lower())
        words = [word for word in clean_title.split() if len(word) > 2]
        title_words = words[:3] if words else ["video"]

        return f"{'-'.join(title_words)}-{video_id}.txt"

    def save_summary(
        self, summary: str, video_id: str, video_title: str = "", video_url: str = ""
    ):
        """Save summary to file with descriptive filename"""
        filename = self.create_descriptive_filename(video_title, video_id)
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(summary)
                if video_url:
                    f.write(f"\n\nSource: {video_url}")
            print(f"\nSummary saved to: {filename}")
        except Exception as e:
            print(f"Error saving summary: {e}")

    def process_video(self, url: str, save_to_file: bool = False):
        """Main processing pipeline"""
        audio_path = None
        video_title = ""
        try:
            # Extract video ID
            video_id = self.extract_video_id(url)

            # Download and extract audio
            audio_path, video_title = self.download_video_audio(url)

            # Transcribe audio
            transcript = self.transcribe_audio(audio_path)

            # Generate summary
            summary = self.generate_summary(transcript)

            print()

            # Print summary (plain text, no separators)
            print(summary)

            # Save to file if requested
            if save_to_file:
                self.save_summary(summary, video_id, video_title, url)

        finally:
            # Clean up temporary files
            if audio_path:
                self.cleanup_temp_files(audio_path)


def main():
    parser = argparse.ArgumentParser(
        description="YouTube Video Summarizer - Generate concise summaries optimized for 4x6 index cards"
    )
    parser.add_argument("url", nargs="?", help="YouTube video URL to summarize")
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the summary to a descriptive filename",
    )
    parser.add_argument(
        "--length",
        choices=["short", "medium", "long"],
        default="short",
        help="Summary length: short (4x6 index card), medium (2-3 paragraphs), long (comprehensive)",
    )

    args = parser.parse_args()

    if not args.url:
        parser.print_usage(sys.stderr)
        sys.exit(2)

    # Validate URL
    if not ("youtube.com" in args.url or "youtu.be" in args.url):
        print("Error: Please provide a valid YouTube URL")
        sys.exit(1)

    # Process video
    summarizer = YtSumm(summary_length=args.length)
    summarizer.process_video(args.url, save_to_file=args.save)


if __name__ == "__main__":
    main()
