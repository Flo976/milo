"""Test MMS-TTS-mlg baseline quality on Malagasy sentences."""
import argparse
import os
import time
import torch
import soundfile as sf
from transformers import VitsModel, AutoTokenizer


TEST_SENTENCES = [
    "Manao ahoana tompoko",
    "Misaotra betsaka",
    "Inona ny vaovao androany",
    "Mila fanampiana aho",
    "Tsara be ny andro androany",
    "Faly mifankahita aminao aho",
    "Aiza ny hopitaly akaiky indrindra",
    "Tiako ny teny malagasy",
    "Mankasitraka anao aho noho ny fanampiana",
    "Mba afaka manampy ahy ve ianao",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/mms-tts-mlg")
    parser.add_argument("--output-dir", default="/home/florent/milo/evaluation/tts_baseline")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    print("=" * 60)
    print("MILO — TTS Baseline Test")
    print("=" * 60)
    print("Model: {}".format(args.model))
    print("Device: {}".format(args.device))

    print("\n[1/3] Loading model and tokenizer...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = VitsModel.from_pretrained(args.model).to(args.device)
    print("  Loaded in {:.1f}s".format(time.time() - t0))
    print("  Parameters: {:.1f}M".format(sum(p.numel() for p in model.parameters()) / 1e6))
    print("  Sample rate: {} Hz".format(model.config.sampling_rate))

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n[2/3] Generating audio for {} sentences...".format(len(TEST_SENTENCES)))
    times = []
    for i, text in enumerate(TEST_SENTENCES):
        t1 = time.time()
        inputs = tokenizer(text, return_tensors="pt").to(args.device)
        with torch.no_grad():
            output = model(**inputs)
        waveform = output.waveform[0].cpu().numpy()
        elapsed = time.time() - t1
        times.append(elapsed)

        duration = len(waveform) / model.config.sampling_rate
        filename = os.path.join(args.output_dir, "{:02d}.wav".format(i + 1))
        sf.write(filename, waveform, model.config.sampling_rate)

        print("  {:02d}. [{:.2f}s gen, {:.2f}s audio] {}".format(
            i + 1, elapsed, duration, text
        ))

    print("\n[3/3] Summary")
    print("=" * 60)
    print("  Sentences: {}".format(len(TEST_SENTENCES)))
    print("  Avg generation time: {:.3f}s".format(sum(times) / len(times)))
    print("  Min: {:.3f}s  Max: {:.3f}s".format(min(times), max(times)))
    print("  Output dir: {}".format(args.output_dir))
    print("=" * 60)
    print("\nListen to the WAV files to evaluate quality.")
    print("Key things to check:")
    print("  - Pronunciation accuracy")
    print("  - Naturalness / prosody")
    print("  - Artifacts or glitches")
    print("  - Speaker consistency")


if __name__ == "__main__":
    main()
