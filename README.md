# yt-summ

yt-summ is a CLI program that pulls YouTube audio with `yt-dlp`, transcribes it locally with Whisper, and feeds the transcript to your preferred LLM for a condensed brief you can archive or diff later.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Highlights

- Download the audio track from a public YouTube video
- Transcribe speech locally using Whisper
- Summarize with Anthropic Claude or OpenAI models
- Choose short, medium, or long summary formats
- Generate descriptive filenames when saving summaries

## Quickstart

```bash
git clone https://github.com/martynasjocius/yt-summ.git
cd yt-summ
python -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```

Set an API key for at least one provider:

```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key"
# or
export OPENAI_API_KEY="your-openai-api-key"
```

## Usage

```bash
./yt-summ "https://www.youtube.com/watch?v=VIDEO_ID"
./yt-summ --save "https://www.youtube.com/watch?v=VIDEO_ID"
./yt-summ --length medium "https://www.youtube.com/watch?v=VIDEO_ID"
./yt-summ --length long --save "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Options

- `--save` writes the summary to disk with an AI-generated filename
- `--length {short,medium,long}` controls the structure and detail level

## Requirements

- Python 3.10+
- API key for Anthropic or OpenAI

## License

MIT License. See [`LICENSE`](LICENSE).
