"""Prepare TTS training data from badrex/malagasy-speech-full.

Loads the dataset, analyzes speakers, selects the best single speaker,
and exports WAV files + metadata CSV for VITS/MMS-TTS fine-tuning.
"""
import argparse
import csv
import os
import re

import numpy as np
import soundfile as sf
from collections import Counter
from datasets import load_dataset, Audio


def normalize_malagasy(text):
    """Normalize Malagasy text for TTS training.

    - Lowercase
    - Remove characters outside the MMS-TTS-mlg vocabulary
    - Collapse multiple spaces
    """
    text = text.lower().strip()
    # MMS-TTS-mlg vocab: a-z (no q,u,w,x), space, apostrophe, hyphen, accented
    # Keep only characters the tokenizer knows
    allowed = set("abcdefghijklmnoprstvy z'-\u00e0\u00ec\u00f2\u00f4\u1ef3")
    # Also keep space
    allowed.add(" ")
    text = "".join(c for c in text if c in allowed)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def analyze_speakers(dataset):
    """Count samples per speaker and return sorted list."""
    speaker_counts = Counter(dataset["speaker_id"])
    sorted_speakers = speaker_counts.most_common()
    return sorted_speakers


def compute_speaker_stats(dataset, speaker_id):
    """Compute duration and quality stats for a given speaker."""
    speaker_ds = dataset.filter(
        lambda x: x["speaker_id"] == speaker_id,
        num_proc=4,
    )
    durations = speaker_ds["audio_duration"]
    return {
        "speaker_id": speaker_id,
        "num_samples": len(speaker_ds),
        "total_duration_min": sum(durations) / 60.0,
        "avg_duration_s": np.mean(durations),
        "min_duration_s": min(durations),
        "max_duration_s": max(durations),
        "median_duration_s": np.median(durations),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare TTS training data for MMS-TTS fine-tuning"
    )
    parser.add_argument(
        "--dataset", default="badrex/malagasy-speech-full",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--output-dir", default="/home/florent/milo/data/tts",
        help="Output directory for WAV files and metadata"
    )
    parser.add_argument(
        "--speaker-id", default=None,
        help="Force a specific speaker_id (skip auto-selection)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=150,
        help="Maximum number of samples to extract (default: 150)"
    )
    parser.add_argument(
        "--min-samples", type=int, default=80,
        help="Minimum samples required from a speaker (default: 80)"
    )
    parser.add_argument(
        "--min-duration", type=float, default=2.0,
        help="Minimum audio duration in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--max-duration", type=float, default=15.0,
        help="Maximum audio duration in seconds (default: 15.0)"
    )
    parser.add_argument(
        "--target-sr", type=int, default=16000,
        help="Target sample rate in Hz (default: 16000, matches MMS-TTS)"
    )
    parser.add_argument(
        "--top-speakers", type=int, default=20,
        help="Number of top speakers to display in analysis (default: 20)"
    )
    parser.add_argument(
        "--split", default="train",
        help="Dataset split to use (default: train)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("MILO -- Prepare TTS Training Data")
    print("=" * 60)
    print("Dataset: {}".format(args.dataset))
    print("Output:  {}".format(args.output_dir))
    print("Target samples: {}-{}".format(args.min_samples, args.max_samples))
    print("Duration filter: {:.1f}s - {:.1f}s".format(args.min_duration, args.max_duration))
    print("Sample rate: {} Hz".format(args.target_sr))
    print("=" * 60)

    # ----------------------------------------------------------------
    # Step 1: Load dataset
    # ----------------------------------------------------------------
    print("\n[1/5] Loading dataset (split='{}')...".format(args.split))
    ds = load_dataset(args.dataset, split=args.split)
    print("  Total samples: {}".format(len(ds)))
    print("  Columns: {}".format(ds.column_names))

    # ----------------------------------------------------------------
    # Step 2: Analyze speakers
    # ----------------------------------------------------------------
    print("\n[2/5] Analyzing speakers...")
    speaker_counts = analyze_speakers(ds)
    total_speakers = len(speaker_counts)
    print("  Total unique speakers: {}".format(total_speakers))
    print("\n  Top {} speakers by sample count:".format(min(args.top_speakers, total_speakers)))
    print("  {:<20s} {:>8s}".format("Speaker ID", "Samples"))
    print("  " + "-" * 30)
    for spk_id, count in speaker_counts[:args.top_speakers]:
        marker = " <--" if count >= args.min_samples else ""
        print("  {:<20s} {:>8d}{}".format(str(spk_id), count, marker))

    eligible = [(s, c) for s, c in speaker_counts if c >= args.min_samples]
    print("\n  Speakers with >= {} samples: {}".format(args.min_samples, len(eligible)))

    # ----------------------------------------------------------------
    # Step 3: Select speaker
    # ----------------------------------------------------------------
    print("\n[3/5] Selecting speaker...")

    if args.speaker_id is not None:
        selected_speaker = args.speaker_id
        selected_count = dict(speaker_counts).get(selected_speaker, 0)
        print("  Forced speaker: {} ({} samples)".format(selected_speaker, selected_count))
        if selected_count < args.min_samples:
            print("  WARNING: Speaker has fewer than {} samples!".format(args.min_samples))
    else:
        if not eligible:
            print("  ERROR: No speaker has >= {} samples.".format(args.min_samples))
            print("  Try lowering --min-samples or using a different dataset split.")
            return

        # Among eligible speakers, pick the one with the most samples.
        # If multiple have similar counts, prefer one with moderate durations
        # (better for TTS training).
        best_speaker = None
        best_score = -1

        for spk_id, count in eligible[:10]:  # Check top 10 eligible
            stats = compute_speaker_stats(ds, spk_id)
            # Score: prefer more samples + moderate average duration (4-8s ideal)
            duration_bonus = 1.0
            avg = stats["avg_duration_s"]
            if 4.0 <= avg <= 8.0:
                duration_bonus = 1.2
            elif 3.0 <= avg <= 10.0:
                duration_bonus = 1.1

            score = count * duration_bonus
            print("  Candidate: {} -- {} samples, avg {:.1f}s, total {:.1f}min (score={:.0f})".format(
                spk_id, count, avg, stats["total_duration_min"], score
            ))

            if score > best_score:
                best_score = score
                best_speaker = spk_id

        selected_speaker = best_speaker
        selected_count = dict(speaker_counts)[selected_speaker]
        print("\n  --> Selected speaker: {} ({} samples)".format(selected_speaker, selected_count))

    # ----------------------------------------------------------------
    # Step 4: Filter and extract samples
    # ----------------------------------------------------------------
    print("\n[4/5] Filtering and extracting samples...")

    # Filter to selected speaker
    speaker_ds = ds.filter(
        lambda x: x["speaker_id"] == selected_speaker,
        num_proc=4,
    )
    print("  Speaker samples (raw): {}".format(len(speaker_ds)))

    # Filter by duration
    speaker_ds = speaker_ds.filter(
        lambda x: args.min_duration <= x["audio_duration"] <= args.max_duration,
        num_proc=4,
    )
    print("  After duration filter ({:.1f}s-{:.1f}s): {}".format(
        args.min_duration, args.max_duration, len(speaker_ds)
    ))

    # Filter out empty or too-short transcriptions
    speaker_ds = speaker_ds.filter(
        lambda x: x["transcription"] is not None and len(x["transcription"].strip()) >= 5,
        num_proc=4,
    )
    print("  After transcription filter: {}".format(len(speaker_ds)))

    # Filter: check that normalized text is not empty
    def has_valid_text(example):
        normalized = normalize_malagasy(example["transcription"])
        return len(normalized) >= 3
    speaker_ds = speaker_ds.filter(has_valid_text, num_proc=4)
    print("  After normalization check: {}".format(len(speaker_ds)))

    if len(speaker_ds) < args.min_samples:
        print("  WARNING: Only {} samples after filtering (wanted >= {}).".format(
            len(speaker_ds), args.min_samples
        ))
        print("  Continuing with available samples...")

    # Sort by duration (prefer moderate lengths) and select
    # We want a good mix: sort by duration, take from the middle range
    durations = speaker_ds["audio_duration"]
    indices = list(range(len(speaker_ds)))
    # Sort by absolute distance from median duration
    median_dur = np.median(durations)
    indices.sort(key=lambda i: abs(durations[i] - median_dur))
    selected_indices = indices[:args.max_samples]
    selected_indices.sort()  # Sort back for consistent ordering

    speaker_ds = speaker_ds.select(selected_indices)
    print("  Final selection: {} samples".format(len(speaker_ds)))

    # Resample audio to target sample rate
    speaker_ds = speaker_ds.cast_column("audio", Audio(sampling_rate=args.target_sr))

    # ----------------------------------------------------------------
    # Step 5: Export WAV files and metadata
    # ----------------------------------------------------------------
    print("\n[5/5] Exporting to {}...".format(args.output_dir))

    wav_dir = os.path.join(args.output_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    metadata_path = os.path.join(args.output_dir, "metadata.csv")
    metadata_rows = []
    durations_exported = []
    skipped = 0

    for i, sample in enumerate(speaker_ds):
        audio_array = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]
        text = normalize_malagasy(sample["transcription"])

        if len(text) < 3:
            skipped += 1
            continue

        filename = "mg_{:04d}.wav".format(i)
        filepath = os.path.join(wav_dir, filename)

        # Ensure float32 and normalize amplitude
        audio_array = np.array(audio_array, dtype=np.float32)
        max_val = np.max(np.abs(audio_array))
        if max_val > 0:
            audio_array = audio_array / max_val * 0.95  # Normalize to 0.95 peak

        sf.write(filepath, audio_array, sr)
        duration = len(audio_array) / sr
        durations_exported.append(duration)

        # Store relative path for metadata
        metadata_rows.append({
            "file_name": "wavs/" + filename,
            "transcription": text,
            "duration": round(duration, 2),
        })

        if (i + 1) % 25 == 0:
            print("  Exported {}/{}...".format(i + 1, len(speaker_ds)))

    # Write metadata CSV (pipe-separated for VITS compatibility)
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerow(["file_name", "transcription", "duration"])
        for row in metadata_rows:
            writer.writerow([row["file_name"], row["transcription"], row["duration"]])

    # Also write a simple filelist format (used by some VITS trainers)
    filelist_path = os.path.join(args.output_dir, "filelist.txt")
    with open(filelist_path, "w", encoding="utf-8") as f:
        for row in metadata_rows:
            f.write("{}|{}\n".format(row["file_name"], row["transcription"]))

    # Write speaker info
    info_path = os.path.join(args.output_dir, "speaker_info.txt")
    with open(info_path, "w", encoding="utf-8") as f:
        f.write("speaker_id={}\n".format(selected_speaker))
        f.write("num_samples={}\n".format(len(metadata_rows)))
        f.write("total_duration_min={:.1f}\n".format(sum(durations_exported) / 60.0))
        f.write("avg_duration_s={:.2f}\n".format(np.mean(durations_exported)))
        f.write("sample_rate={}\n".format(args.target_sr))
        f.write("skipped={}\n".format(skipped))

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)
    print("  Speaker:        {}".format(selected_speaker))
    print("  Samples:        {}".format(len(metadata_rows)))
    print("  Skipped:        {}".format(skipped))
    print("  Total duration: {:.1f} min".format(sum(durations_exported) / 60.0))
    print("  Avg duration:   {:.2f}s".format(np.mean(durations_exported)))
    print("  Sample rate:    {} Hz".format(args.target_sr))
    print("  WAV directory:  {}".format(wav_dir))
    print("  Metadata CSV:   {}".format(metadata_path))
    print("  Filelist:       {}".format(filelist_path))
    print("  Speaker info:   {}".format(info_path))
    print("=" * 60)

    if len(metadata_rows) < args.min_samples:
        print("\nWARNING: Got {} samples, which is below the {} minimum.".format(
            len(metadata_rows), args.min_samples
        ))
        print("The fine-tuning may still work but quality could be lower.")
    elif len(metadata_rows) >= args.min_samples:
        print("\nReady for fine-tuning! Run 07_finetune_tts.py next.")


if __name__ == "__main__":
    main()
