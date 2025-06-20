# Kokoro TTS Test Script

This repository contains a test script for running text-to-speech (TTS) synthesis using the Kokoro library, with support for Spanish and English voices. It is optimized for Apple Silicon (M1/M2/M3/M4) and will use the GPU (MPS) if available.

## Features
- Synthesize speech from text using Kokoro and save as WAV files
- Supports Spanish (Dora) and English voices
- Benchmarks memory and time usage
- Automatically uses Apple Silicon GPU (MPS) if available
- Cleans up all test/produced WAV files via `.gitignore`

## Usage
1. Install dependencies:
   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   pip install kokoro soundfile psutil
   ```
2. Run the test script:
   ```sh
   python basic_test.py
   ```

## Benchmarks (Apple Silicon M4, macOS, PyTorch MPS)

| Device | Synthesis Time | Peak Memory Usage |
|--------|----------------|------------------|
| CPU    | ~2.23 s        | ~2.6 GB          |
| MPS    | ~1.89 s        | ~1.3 GB          |

- **MPS (Apple GPU) is faster and uses less memory than CPU for this TTS task.**
- Output files are identical in quality.

## Notes
- All generated `.wav` files are ignored by git.
- For best performance on Apple Silicon, ensure you have a recent version of PyTorch with MPS support.
- For longer texts, split into smaller chunks to avoid model context length errors.

## License
MIT
