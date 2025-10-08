# YTVidSum

A Python tool that downloads YouTube videos, transcribes their audio content, and generates concise summaries using AI.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Transform any YouTube video into a concise summary perfect for note-taking and quick reference.

## Features

- 🎥 **YouTube Integration**: Downloads audio from any YouTube video
- 🎤 **Audio Transcription**: Uses OpenAI Whisper for accurate speech-to-text
- 🤖 **AI Summarization**: Leverages Anthropic Claude or OpenAI for intelligent summarization
- 📝 **Concise Format**: Generates summaries in various lengths (short/medium/long)
- 🏷️ **Smart Filenames**: Uses AI to generate descriptive, short filenames instead of word collections
- ⚡ **Fast & Efficient**: Quick processing with minimal dependencies

## Installation

1. Clone the repository:

```bash
git clone https://github.com/martynasjocius/ytvidsum.git
cd ytvidsum
```

2. Install dependencies:

```bash
virtualenv venv
. ./venv/bin/activate
pip install -r requirements.txt
```

3. Set up your API key:

```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Alternative: OpenAI
export OPENAI_API_KEY="your-openai-api-key"
```

## Usage

```bash
# Basic usage (short summary, no file save)
./ytvidsum.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Save to file with descriptive filename
./ytvidsum.py --save "https://www.youtube.com/watch?v=VIDEO_ID"

# Different summary lengths
./ytvidsum.py --length medium "https://www.youtube.com/watch?v=VIDEO_ID"
./ytvidsum.py --length long --save "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Options

- `--save`: Save summary to file with AI-generated descriptive filename (e.g., `homelab-server-setup-KaHuP8nDnuY.txt`)
- `--length {short,medium,long}`: Summary length (default: short)
  - **short**: Concise format with bullet points
  - **medium**: 2-3 paragraphs or well-structured list
  - **long**: Comprehensive summary with headings

The script will:

1. Download the video's audio track
2. Transcribe the audio to text
3. Generate a summary using AI
4. Display the summary in your terminal
5. Save to file with source URL (if `--save` specified)

## Example Output

```
Video title: How to Learn Python in 2024
Transcribing audio...
Generating summary...

PYTHON PROGRAMMING
• **Core Concepts**: Variables, functions, loops, data structures
• **Best Resources**: Python.org tutorial, Real Python, Codecademy
• **Practice Projects**: Build calculator, web scraper, data analyzer
• **Key Libraries**: NumPy (data), Pandas (analysis), Flask (web)
• **Next Steps**: Contribute to open source, build portfolio

Summary saved to: python-beginners-guide-VIDEO_ID.txt
```

**Saved file contents:**
```
PYTHON PROGRAMMING
• **Core Concepts**: Variables, functions, loops, data structures
• **Best Resources**: Python.org tutorial, Real Python, Codecademy
• **Practice Projects**: Build calculator, web scraper, data analyzer
• **Key Libraries**: NumPy (data), Pandas (analysis), Flask (web)
• **Next Steps**: Contribute to open source, build portfolio

Source: https://www.youtube.com/watch?v=VIDEO_ID
```

## Requirements

- Python 3.8+
- Internet connection
- Anthropic or OpenAI API key

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

**Martynas Jocius** - 2025

