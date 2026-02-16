"""Evaluate a fine-tuned Whisper model on the Malagasy test set."""
import argparse
import json
import os
import time
import torch
from transformers import pipeline, WhisperProcessor
from datasets import load_dataset
from jiwer import wer, cer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to fine-tuned model")
    parser.add_argument("--dataset", default="badrex/malagasy-speech-full")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", default="/home/florent/milo/evaluation/results.json")
    args = parser.parse_args()

    print("=" * 60)
    print("MILO — Evaluation")
    print("=" * 60)
    print("Model: {}".format(args.model))
    print("Split: {}".format(args.split))

    processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
    pipe = pipeline(
        "automatic-speech-recognition",
        model=args.model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device="cuda:0",
        torch_dtype=torch.float16,
    )

    ds = load_dataset(args.dataset, split=args.split)
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    print("Samples: {}".format(len(ds)))

    predictions = []
    references = []
    details = []
    t0 = time.time()

    for i, sample in enumerate(ds):
        audio = sample["audio"]["array"]
        ref = sample["transcription"].strip()
        pred = pipe(
            audio,
            return_timestamps=True,
            generate_kwargs={"language": "mg", "task": "transcribe"},
        )["text"].strip()

        predictions.append(pred)
        references.append(ref)

        sample_wer = wer(ref.lower(), pred.lower())
        sample_cer = cer(ref.lower(), pred.lower())
        details.append({"ref": ref, "pred": pred, "wer": sample_wer, "cer": sample_cer})

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            speed = (i + 1) / elapsed
            eta = (len(ds) - i - 1) / speed
            print("  {}/{} ({:.0f}/s, ETA {:.0f}s)".format(i + 1, len(ds), speed, eta))

    avg_wer = wer(
        [r.lower() for r in references],
        [p.lower() for p in predictions],
    )
    avg_cer = cer(
        [r.lower() for r in references],
        [p.lower() for p in predictions],
    )

    elapsed = time.time() - t0

    results = {
        "model": args.model,
        "split": args.split,
        "num_samples": len(ds),
        "wer": round(avg_wer * 100, 2),
        "cer": round(avg_cer * 100, 2),
        "eval_time_s": round(elapsed, 1),
    }

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print("WER: {:.1f}%".format(results["wer"]))
    print("CER: {:.1f}%".format(results["cer"]))
    print("Time: {:.0f}s".format(elapsed))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"summary": results, "details": details[:50]}, f, indent=2, ensure_ascii=False)
    print("Saved to {}".format(args.output))

    # Also write a simple status file for the orchestrator
    status_path = os.path.dirname(args.output) + "/eval_status.txt"
    with open(status_path, "w") as f:
        f.write("WER={:.1f}\nCER={:.1f}\nSAMPLES={}\nMODEL={}\n".format(
            results["wer"], results["cer"], results["num_samples"], args.model
        ))

    return results


if __name__ == "__main__":
    main()
