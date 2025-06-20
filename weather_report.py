# Requirements: pip install requests soundfile numpy torch psutil
# Also requires: Kokoro TTS, Ollama running with qwen3:0.6b pulled, mpv (for playback)

import os
import requests
import json
import soundfile as sf
import numpy as np
import torch
import kokoro
import time
import psutil
from text_splitter import split_text_for_tts, safe_tts_chunks
from datetime import datetime, timedelta, timezone

def get_weather(api_key, city="Montevideo,UY"):
    """Fetch weather data from OpenWeatherMap for a given city (default: Montevideo,UY)."""
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric&lang=es"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()

def format_decimal(value):
    """Format a float as Spanish text with 'punto' instead of dot for decimals."""
    s = f"{value:.2f}".replace('.', ' punto ')
    # Remove trailing '00' if integer
    if s.endswith(' punto 00'):
        s = s[:-8]
    return s

def format_timestamp(ts, tz_offset):
    """Format a Unix timestamp and timezone offset as a Spanish datetime string for Montevideo."""
    dt = datetime.utcfromtimestamp(ts) + timedelta(seconds=tz_offset)
    return dt.strftime('%H:%M:%S del %d/%m/%Y')

def ollama_weather_report(weather_json):
    """Generate a Spanish weather report using Ollama LLM from weather JSON, with no markup/tags."""
    tz_offset = weather_json.get('timezone', 0)
    dt_str = format_timestamp(weather_json['dt'], tz_offset)
    sunrise_str = format_timestamp(weather_json['sys']['sunrise'], tz_offset)
    sunset_str = format_timestamp(weather_json['sys']['sunset'], tz_offset)
    temp = format_decimal(weather_json['main']['temp'])
    feels_like = format_decimal(weather_json['main']['feels_like'])
    temp_min = format_decimal(weather_json['main']['temp_min'])
    temp_max = format_decimal(weather_json['main']['temp_max'])
    humidity = weather_json['main']['humidity']
    wind_speed = format_decimal(weather_json['wind']['speed'])
    clouds = weather_json['clouds']['all']
    city = weather_json['name']
    country = weather_json['sys']['country']
    weather_desc = weather_json['weather'][0]['description']
    visibility = weather_json.get('visibility', None)
    prompt = (
        f"Eres una meteoróloga uruguaya joven y simpática. Escribe un informe del tiempo completo pero breve en español, "
        f"usando exclusivamente los siguientes datos ya preformateados para Montevideo, Uruguay. Sé clara, fresca, natural y un poco conversacional, como si hablaras con amigos o familia. "
        f"Cuando menciones números decimales, usa la palabra 'punto' en vez del símbolo. "
        f"Haz el informe más interesante y útil para el público general, agregando explicaciones o consejos prácticos sobre el clima, pero sin inventar datos. "
        f"Hora local actual: {dt_str}. "
        f"Salida del sol: {sunrise_str}. "
        f"Puesta del sol: {sunset_str}. "
        f"Temperatura actual: {temp} grados Celsius. "
        f"Sensación térmica: {feels_like} grados. "
        f"Temperatura mínima: {temp_min} grados, máxima: {temp_max} grados. "
        f"Humedad: {humidity} por ciento. "
        f"Viento: {wind_speed} kilómetros por hora. "
        f"Nubosidad: {clouds} por ciento. "
        f"Condición principal: {weather_desc}. "
        + (f"Visibilidad: {visibility} metros. " if visibility is not None else "")
        + "No uses ningún tipo de marcado, etiquetas, ni formato especial: solo texto plano, ya que el resultado será leído por un sistema TTS. "
        + "No inventes ni asumas datos que no estén explícitamente presentes arriba. "
        + "Redacta el informe de forma natural y humana, explicando el significado de los valores para el público general."
    )
    ollama_url = "http://localhost:11434/api/generate"
    data = {
        "model": "gemma3:4b",
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(ollama_url, json=data, timeout=60)
        response.raise_for_status()
        raw = response.json()["response"].strip()
        # Remove <think>...</think> blocks if present
        import re
        cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL|re.IGNORECASE).strip()
        return cleaned
    except requests.exceptions.ConnectionError:
        print("[ERROR] Ollama server is not running at http://localhost:11434. Please start Ollama.")
        raise
    except Exception as e:
        print(f"[ERROR] Ollama request failed: {e}")
        raise

def select_device():
    """Select the best available device: CUDA, MPS, or CPU."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def tts_kokoro(text, lang="es", device=None):
    """Synthesize text to speech using Kokoro TTS, handling long texts and benchmarking."""
    if device is None:
        device = select_device()
    from kokoro import KModel, KPipeline
    voice = "ef_dora"  # Use a Spanish voice as in basic_test.py
    model = KModel().to(device).eval()
    pipeline = KPipeline(lang_code='e', model=model)
    pack = pipeline.load_voice(voice)
    t0 = time.time()
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss
    audio_segments = safe_tts_chunks(pipeline, model, pack, text, voice)
    total_len = sum(len(seg) for seg in audio_segments)
    mem_after = process.memory_info().rss
    t1 = time.time()
    print(f"Kokoro TTS: {total_len/24000:.2f}s audio, time: {t1-t0:.2f}s, mem: {(mem_after-mem_before)/1e6:.1f}MB, device: {device}")
    out_path = "weather_report_es.wav"
    if audio_segments:
        full_audio = np.concatenate(audio_segments)
        sf.write(out_path, full_audio, 24000)
        print(f"Audio saved to {out_path}")
        return out_path
    else:
        print("No audio generated.")
        return None

def main():
    """Main pipeline: fetch weather, generate report, synthesize audio, and play it."""
    api_key = os.environ.get("OPENWEATHER_KEY")
    if not api_key:
        raise RuntimeError("OPENWEATHER_KEY environment variable not set.")
    weather = get_weather(api_key)
    print("Weather JSON:", json.dumps(weather, indent=2, ensure_ascii=False))
    report = ollama_weather_report(weather)
    print("\nWeather report (Spanish):\n", report)
    audio_path = tts_kokoro(report, lang="es")
    print(f"Audio file: {audio_path}")
    # Optionally, play audio (requires mpv)
    if audio_path:
        try:
            exit_code = os.system(f"mpv --no-terminal {audio_path} > /dev/null 2>&1")
            if exit_code != 0:
                print("[INFO] Could not play audio: mpv not found or playback failed.")
        except Exception as e:
            print(f"[INFO] Could not play audio: {e}")

if __name__ == "__main__":
    main()
