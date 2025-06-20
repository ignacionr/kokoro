import re
import torch
import numpy as np

def split_text_for_tts(text, max_length=150):
    """
    Splits text into chunks suitable for TTS synthesis.
    Tries to split at sentence boundaries, but will further split long sentences if needed.
    """
    # First, split by sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ''
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_length:
            current = (current + ' ' + sentence).strip()
        else:
            if current:
                chunks.append(current)
            if len(sentence) > max_length:
                # Further split long sentences
                for i in range(0, len(sentence), max_length):
                    chunks.append(sentence[i:i+max_length])
                current = ''
            else:
                current = sentence
    if current:
        chunks.append(current)
    return [c.strip() for c in chunks if c.strip()]

def safe_tts_chunks(pipeline, model, pack, text, voice):
    """
    Synthesize TTS audio for text, splitting further if a chunk is too long for the model.
    """
    audio_segments = []
    for chunk in split_text_for_tts(text):
        try:
            ps = None
            for _, ps_candidate, _ in pipeline(chunk, voice):
                ps = ps_candidate
                break
            if isinstance(ps, tuple):
                ps_for_model = ps[0]
            else:
                ps_for_model = ps
            idx = min(len(ps_for_model)-1, len(pack)-1) if hasattr(ps_for_model, '__len__') else len(pack)-1
            ref_s = pack[idx]
            with torch.no_grad():
                audio = model(ps_for_model, ref_s, speed=1)
            if hasattr(audio, 'detach'):
                audio_np = audio.detach().cpu().numpy()
            else:
                audio_np = np.array(audio)
            audio_segments.append(audio_np)
        except AssertionError as e:
            # If chunk is still too long, split by half and retry
            if len(chunk) > 50:
                mid = len(chunk) // 2
                left = chunk[:mid]
                right = chunk[mid:]
                audio_segments.extend(safe_tts_chunks(pipeline, model, pack, left, voice))
                audio_segments.extend(safe_tts_chunks(pipeline, model, pack, right, voice))
            else:
                print(f"[TTS ERROR] Could not synthesize chunk: {chunk[:30]}... (len={len(chunk)})")
    return audio_segments
