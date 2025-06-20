from kokoro import KModel, KPipeline
import soundfile as sf
import numpy as np
import subprocess
import torch
import os
import psutil
import time

# Function to inspect and print type and structure of ps
def inspect_ps(ps, lang):
    print(f"\n--- {lang} phoneme sequence inspection ---")
    print(f"Type: {type(ps)}")
    if isinstance(ps, (list, tuple)):
        print(f"Length: {len(ps)}")
        for i, item in enumerate(ps):
            print(f"  [{i}] type: {type(item)}, value: {item}")
    else:
        print(f"Value: {ps}")
    print("-------------------------------\n")

def print_memory_usage(stage):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"[MEMORY] {stage}: {mem_mb:.2f} MB")

def run_synthesis_mps():
    print(f"\n===== Running on device: mps =====")
    print_memory_usage(f"Start (mps)")
    model = KModel().to('mps').eval()
    print(f"Model device: {next(model.parameters()).device}")
    print_memory_usage(f"After model load (mps)")
    pipeline_es = KPipeline(lang_code='e', model=model)
    print_memory_usage(f"After pipeline init (mps)")
    pack_es = pipeline_es.load_voice('ef_dora')
    print_memory_usage(f"After voice load (mps)")
    text_es = (
        "Hola, mi nombre es Dora. Esta es una prueba de voz en español. "
        "Voy a leer un texto largo para comprobar la capacidad del sistema de síntesis de voz Kokoro. "
        "La inteligencia artificial ha avanzado mucho en los últimos años. "
        "Ahora, los sistemas de texto a voz pueden generar discursos completos o ayudar a personas con discapacidades visuales. "
        "Gracias por probar Kokoro."
    )
    ps_es = None
    for _, ps_candidate, _ in pipeline_es(text_es, 'ef_dora'):
        ps_es = ps_candidate
        break
    inspect_ps(ps_es, 'Spanish (mps)')
    print_memory_usage(f"After tokenization (mps)")
    # Try workaround: if ps is a tuple, use only the first element
    if isinstance(ps_es, tuple):
        ps_for_model = ps_es[0]
    else:
        ps_for_model = ps_es
    if ps_for_model is not None:
        idx = min(len(ps_for_model)-1, len(pack_es)-1) if hasattr(ps_for_model, '__len__') else len(pack_es)-1
        ref_s_es = pack_es[idx]
        start = time.time()
        with torch.no_grad():
            audio_es = model(ps_for_model, ref_s_es, speed=1)
        elapsed = time.time() - start
        print(f"[TIME] Synthesis time on mps: {elapsed:.2f} seconds")
        print_memory_usage(f"After synthesis (mps)")
        if hasattr(audio_es, 'detach'):
            audio_np_es = audio_es.detach().cpu().numpy()
        else:
            audio_np_es = np.array(audio_es)
        outname = "spanish_dora_shorter_mps.wav"
        sf.write(outname, audio_np_es, 24000)
        print(f"Saved: {outname}")
        subprocess.run(["mpv", outname])
        print_memory_usage(f"After save/playback (mps)")

# Only run on MPS
mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
if mps_available:
    run_synthesis_mps()
else:
    print("MPS (Apple Silicon GPU) is not available on this system.")