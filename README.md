# YTVidSum

Turn any YouTube link into a transcript and an AI-generated summary that is ready for quick review or note taking.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Highlights

- Download the audio track from a public YouTube video
- Transcribe speech locally using Whisper
- Summarize with Anthropic Claude or OpenAI models
- Choose short, medium, or long summary formats
- Generate descriptive filenames when saving summaries

## Quickstart

```bash
git clone https://github.com/martynasjocius/ytvidsum.git
cd ytvidsum
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
./ytvidsum.py "https://www.youtube.com/watch?v=VIDEO_ID"
./ytvidsum.py --save "https://www.youtube.com/watch?v=VIDEO_ID"
./ytvidsum.py --length medium "https://www.youtube.com/watch?v=VIDEO_ID"
./ytvidsum.py --length long --save "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Options

- `--save` writes the summary to disk with an AI-generated filename
- `--length {short,medium,long}` controls the structure and detail level

## What to Expect

1. Audio is downloaded from the video
2. Whisper creates a transcript
3. The selected model produces a summary
4. The summary is printed to the terminal and, if requested, saved alongside the original URL

### Example output

```
Video title: How to Learn Python in 2024
Transcribing audio...
Generating summary...

PYTHON PROGRAMMING
• Core concepts: variables, functions, loops, data structures
• Best resources: python.org tutorial, Real Python, Codecademy
• Practice projects: build calculator, web scraper, data analyzer
• Key libraries: NumPy (data), Pandas (analysis), Flask (web)
• Next steps: contribute to open source, build portfolio

Summary saved to: python-beginners-guide-VIDEO_ID.txt
```

## Requirements

- Python 3.8+
- Internet access
- API key for Anthropic or OpenAI

## Contributing

Pull requests and issue reports are welcome.

## License

MIT License. See [`LICENSE`](LICENSE).
