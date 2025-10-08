#!/usr/bin/env python3
#
# YTVidSum - YouTube Video Summarizer
#
# Downloads YouTube video content, transcribes audio, and generates a concise summary
# using LLM with different length options (short/medium/long).
#
# Copyright (c) 2025 Martynas Jocius
#

import argparse
import os
import sys
import tempfile
from pathlib import Path

try:
    import yt_dlp
    import whisper
    import anthropic
    import openai
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Please install required packages:")
    print("pip install yt-dlp openai-whisper anthropic openai")
    sys.exit(1)


class YTVidSum:
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

    def _get_video_title(self, url: str) -> str:
        """Get video title without downloading"""
        try:
            ydl_opts = {
                "quiet": True,
                "no_warnings": True,
                "cookiesfrombrowser": (
                    "chromium",
                    "chrome",
                    "firefox",
                    "brave",
                    "edge",
                ),
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return info.get("title", "Unknown")
        except Exception:
            return "Unknown"

    def download_video_audio(self, url: str) -> tuple[str, str]:
        """Download video and extract audio"""
        print(f"Downloading video from: {url}")

        # Create a temporary directory that won't be auto-deleted
        temp_dir = tempfile.mkdtemp()

        try:
            # Configure yt-dlp options
            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": os.path.join(temp_dir, "%(title)s.%(ext)s"),
                "extractaudio": True,
                "audioformat": "wav",
                "noplaylist": True,
                # Add cookie support for age-restricted videos
                "cookiesfrombrowser": ("chrome",),  # Try Chrome first
                "extractor_retries": 3,
                "fragment_retries": 3,
            }

            # Try different browsers for cookie support
            browsers = ["chromium", "chrome", "firefox", "brave", "edge"]

            for browser in browsers:
                try:
                    ydl_opts["cookiesfrombrowser"] = (browser,)
                    print(f"Trying with {browser} cookies...")

                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(url, download=True)
                        video_title = info.get("title", "Unknown")
                        print(f"Video title: {video_title}")

                        # Find the downloaded audio file
                        audio_files = list(Path(temp_dir).glob("*.wav"))
                        if not audio_files:
                            # Try other audio formats
                            audio_files = (
                                list(Path(temp_dir).glob("*.m4a"))
                                + list(Path(temp_dir).glob("*.mp3"))
                                + list(Path(temp_dir).glob("*.webm"))
                            )

                        if not audio_files:
                            raise Exception("No audio file found after download")

                        audio_path = str(audio_files[0])
                        print(f"Audio file: {audio_path}")
                        return audio_path, video_title

                except Exception as e:
                    # Check if we actually have a downloaded file despite cookie errors
                    audio_files = (
                        list(Path(temp_dir).glob("*.wav"))
                        + list(Path(temp_dir).glob("*.m4a"))
                        + list(Path(temp_dir).glob("*.mp3"))
                        + list(Path(temp_dir).glob("*.webm"))
                    )

                    if audio_files:
                        # Download was successful, just cookie extraction failed
                        audio_path = str(audio_files[0])
                        print(f"Download successful! Audio file: {audio_path}")
                        return audio_path, video_title

                    if browser == browsers[-1]:  # Last browser attempt
                        print(f"Failed with all browsers: {e}")
                        print("Trying without cookies...")

                        # Try without cookies as last resort
                        try:
                            ydl_opts_no_cookies = {
                                "format": "bestaudio/best",
                                "outtmpl": os.path.join(temp_dir, "%(title)s.%(ext)s"),
                                "extractaudio": True,
                                "audioformat": "wav",
                                "noplaylist": True,
                                "extractor_retries": 3,
                                "fragment_retries": 3,
                            }

                            with yt_dlp.YoutubeDL(ydl_opts_no_cookies) as ydl:
                                info = ydl.extract_info(url, download=True)
                                video_title = info.get("title", "Unknown")
                                print(f"Video title: {video_title}")

                                # Find the downloaded audio file
                                audio_files = list(Path(temp_dir).glob("*.wav"))
                                if not audio_files:
                                    # Try other audio formats
                                    audio_files = (
                                        list(Path(temp_dir).glob("*.m4a"))
                                        + list(Path(temp_dir).glob("*.mp3"))
                                        + list(Path(temp_dir).glob("*.webm"))
                                    )

                                if not audio_files:
                                    raise Exception(
                                        "No audio file found after download"
                                    )

                                audio_path = str(audio_files[0])
                                print(f"Audio file: {audio_path}")
                                return audio_path, video_title

                        except Exception as e2:
                            print(f"Error downloading video: {e2}")
                            print("\nTroubleshooting tips:")
                            print(
                                "1. Make sure you're logged into YouTube in your browser"
                            )
                            print(
                                "2. Try accessing the video directly in your browser first"
                            )
                            print(
                                "3. Some videos may be region-restricted or require special permissions"
                            )
                            print(
                                "4. Try using a different browser or clearing browser data"
                            )
                            sys.exit(1)
                    else:
                        print(
                            f"Cookie extraction failed for {browser}, trying next browser..."
                        )
                        continue
        except Exception as e:
            # Clean up temp directory on error
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)
            raise e

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

    def cleanup_temp_files(self, audio_path: str):
        """Clean up temporary files"""
        try:
            import shutil

            temp_dir = os.path.dirname(audio_path)
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass  # Ignore cleanup errors

    def extract_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL"""
        import re

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
        import time

        return f"video_{int(time.time())}"

    def create_descriptive_filename(self, video_title: str, video_id: str) -> str:
        """Create a descriptive filename from video title and ID"""
        import re

        # Clean the title: remove special characters, convert to lowercase
        clean_title = re.sub(r"[^\w\s-]", "", video_title.lower())

        # Split into words and filter out common words
        words = clean_title.split()
        common_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "can",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
            "my",
            "your",
            "his",
            "her",
            "its",
            "our",
            "their",
            "how",
            "what",
            "when",
            "where",
            "why",
            "who",
            "which",
        }

        # Filter out common words and keep meaningful ones
        meaningful_words = [
            word for word in words if word not in common_words and len(word) > 2
        ]

        # Take first 3 meaningful words (or all if less than 3)
        title_words = meaningful_words[:3]

        # Join with hyphens and add video ID
        if title_words:
            filename = f"{'-'.join(title_words)}-{video_id}.txt"
        else:
            # Fallback if no meaningful words found
            filename = f"video-{video_id}.txt"

        return filename

    def save_summary(self, summary: str, video_id: str, video_title: str = ""):
        """Save summary to file with descriptive filename"""
        filename = self.create_descriptive_filename(video_title, video_id)
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(summary)
            print(f"Summary saved to: {filename}")
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
                self.save_summary(summary, video_id, video_title)

        finally:
            # Clean up temporary files
            if audio_path:
                self.cleanup_temp_files(audio_path)


def main():
    parser = argparse.ArgumentParser(
        description="YouTube Video Summarizer - Generate concise summaries optimized for 4x6 index cards"
    )
    parser.add_argument("url", help="YouTube video URL to summarize")
    parser.add_argument(
        "--save", action="store_true", help="Save summary to file named VIDEO_ID.txt"
    )
    parser.add_argument(
        "--length",
        choices=["short", "medium", "long"],
        default="short",
        help="Summary length: short (4x6 index card), medium (2-3 paragraphs), long (comprehensive)",
    )

    args = parser.parse_args()

    # Validate URL
    if not ("youtube.com" in args.url or "youtu.be" in args.url):
        print("Error: Please provide a valid YouTube URL")
        sys.exit(1)

    # Validate length argument
    if args.length not in ["short", "medium", "long"]:
        print("Error: --length must be one of: short, medium, long")
        sys.exit(1)

    # Process video
    summarizer = YTVidSum(summary_length=args.length)
    summarizer.process_video(args.url, save_to_file=args.save)


if __name__ == "__main__":
    main()
