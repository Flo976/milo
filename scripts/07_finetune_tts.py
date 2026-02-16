"""Fine-tune facebook/mms-tts-mlg on single-speaker Malagasy data.

Uses the approach from ylacombe/finetune-hf-vits:
1. Clone the finetune-hf-vits repo (if not already cloned)
2. Convert the discriminator checkpoint for Malagasy (mlg)
3. Check tokenizer coverage and extend if needed
4. Generate training config JSON
5. Launch training via accelerate

Requirements (install in order):
    pip install transformers>=4.35.1 datasets[audio]>=2.14.7 accelerate>=0.24.1
    pip install matplotlib tensorboard Cython soundfile scipy

The fine-tuning repo (ylacombe/finetune-hf-vits) also needs:
    - monotonic_align (built from the repo with Cython)
    - The custom utils module from the repo
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import csv


# ---------------------------------------------------------------------------
# MMS-TTS-mlg vocab (from facebook/mms-tts-mlg/vocab.json)
# ---------------------------------------------------------------------------
MMS_MLG_VOCAB = set(
    "abcdefghijklmnoprstvy z'-\u00e0\u00ec\u00f2\u00f4\u1ef3|"
)


def run_cmd(cmd, cwd=None, check=True):
    """Run a shell command and return stdout."""
    print("  $ {}".format(cmd))
    result = subprocess.run(
        cmd, shell=True, cwd=cwd,
        capture_output=True, text=True,
    )
    if result.stdout.strip():
        for line in result.stdout.strip().split("\n")[:20]:
            print("    {}".format(line))
    if result.returncode != 0 and check:
        print("  STDERR: {}".format(result.stderr.strip()[:500]))
        raise RuntimeError("Command failed with code {}".format(result.returncode))
    return result


def ensure_repo_cloned(repo_dir):
    """Clone ylacombe/finetune-hf-vits if not present."""
    if os.path.isdir(os.path.join(repo_dir, ".git")):
        print("  Repo already cloned at {}".format(repo_dir))
        return True

    print("  Cloning ylacombe/finetune-hf-vits...")
    parent = os.path.dirname(repo_dir)
    os.makedirs(parent, exist_ok=True)
    run_cmd(
        "git clone https://github.com/ylacombe/finetune-hf-vits.git {}".format(repo_dir)
    )
    return True


def ensure_requirements(repo_dir):
    """Install requirements from the repo."""
    req_file = os.path.join(repo_dir, "requirements.txt")
    if os.path.isfile(req_file):
        print("  Installing requirements...")
        run_cmd("{} -m pip install -r {}".format(sys.executable, req_file), check=False)
    else:
        print("  No requirements.txt found, installing manually...")
        run_cmd(
            "{} -m pip install transformers>=4.35.1 datasets[audio]>=2.14.7 "
            "accelerate>=0.24.1 matplotlib Cython tensorboard".format(sys.executable),
            check=False,
        )


def build_monotonic_align(repo_dir):
    """Build the monotonic_align Cython extension."""
    ma_dir = os.path.join(repo_dir, "monotonic_align")
    if not os.path.isdir(ma_dir):
        print("  WARNING: monotonic_align directory not found in repo.")
        return False

    # Check if already built
    so_files = [f for f in os.listdir(ma_dir) if f.endswith(".so") or f.endswith(".pyd")]
    inner_dir = os.path.join(ma_dir, "monotonic_align")
    if os.path.isdir(inner_dir):
        so_inner = [f for f in os.listdir(inner_dir) if f.endswith(".so") or f.endswith(".pyd")]
        if so_inner:
            print("  monotonic_align already built.")
            return True

    print("  Building monotonic_align Cython extension...")
    os.makedirs(inner_dir, exist_ok=True)
    run_cmd(
        "{} setup.py build_ext --inplace".format(sys.executable),
        cwd=ma_dir,
        check=False,
    )
    return True


def convert_discriminator(repo_dir, lang_code, output_dir):
    """Convert the MMS discriminator checkpoint for the given language."""
    checkpoint_dir = os.path.join(output_dir, "converted_checkpoint")

    # Check if already converted
    config_file = os.path.join(checkpoint_dir, "config.json")
    if os.path.isfile(config_file):
        print("  Discriminator checkpoint already converted at {}".format(checkpoint_dir))
        return checkpoint_dir

    print("  Converting discriminator for language '{}'...".format(lang_code))
    os.makedirs(checkpoint_dir, exist_ok=True)

    convert_script = os.path.join(repo_dir, "convert_original_discriminator_checkpoint.py")
    if not os.path.isfile(convert_script):
        raise FileNotFoundError(
            "convert_original_discriminator_checkpoint.py not found at {}".format(convert_script)
        )

    run_cmd(
        "{} {} --language_code {} --pytorch_dump_folder_path {}".format(
            sys.executable, convert_script, lang_code, checkpoint_dir
        ),
        cwd=repo_dir,
    )

    if not os.path.isfile(config_file):
        raise RuntimeError(
            "Discriminator conversion failed: no config.json in {}".format(checkpoint_dir)
        )

    print("  Discriminator converted successfully.")
    return checkpoint_dir


def check_tokenizer_coverage(checkpoint_dir, data_dir):
    """Check if all characters in the training data are in the tokenizer vocab."""
    vocab_path = os.path.join(checkpoint_dir, "vocab.json")
    if not os.path.isfile(vocab_path):
        print("  WARNING: vocab.json not found, skipping coverage check.")
        return set()

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    vocab_chars = set(vocab.keys())

    # Read all transcriptions from metadata
    metadata_path = os.path.join(data_dir, "metadata.csv")
    all_chars = set()
    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        header = next(reader)
        for row in reader:
            if len(row) >= 2:
                text = row[1]
                all_chars.update(set(text))

    missing = all_chars - vocab_chars
    if missing:
        print("  WARNING: {} characters in data not in vocab: {}".format(
            len(missing), missing
        ))
        print("  These characters will be treated as <unk> by the tokenizer.")
        print("  Consider extending the tokenizer or further normalizing the data.")
    else:
        print("  All characters in training data are covered by the tokenizer.")

    return missing


def prepare_hf_dataset(data_dir, target_sr):
    """Create a HuggingFace dataset from the prepared data for use with finetune-hf-vits.

    The ylacombe repo expects a HuggingFace dataset with 'audio' and 'transcription' columns.
    We save this locally so the training script can load it.
    """
    from datasets import Dataset, Audio as AudioFeature

    metadata_path = os.path.join(data_dir, "metadata.csv")
    rows = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        header = next(reader)
        for row in reader:
            if len(row) >= 2:
                wav_path = os.path.join(data_dir, row[0])
                rows.append({
                    "audio": wav_path,
                    "transcription": row[1],
                })

    ds = Dataset.from_dict({
        "audio": [r["audio"] for r in rows],
        "transcription": [r["transcription"] for r in rows],
    })
    ds = ds.cast_column("audio", AudioFeature(sampling_rate=target_sr))

    hf_ds_path = os.path.join(data_dir, "hf_dataset")
    ds.save_to_disk(hf_ds_path)
    print("  HuggingFace dataset saved to {}".format(hf_ds_path))
    print("  {} samples, columns: {}".format(len(ds), ds.column_names))
    return hf_ds_path


def generate_training_config(
    checkpoint_dir,
    data_dir,
    output_dir,
    num_epochs,
    batch_size,
    learning_rate,
    eval_steps,
    sample_text,
    target_sr,
):
    """Generate a JSON training config for run_vits_finetuning.py."""

    hf_ds_path = os.path.join(data_dir, "hf_dataset")

    config = {
        "project_name": "mms_mlg_finetuning",
        "push_to_hub": False,
        "report_to": ["tensorboard"],
        "output_dir": output_dir,

        # Dataset — load from local HF dataset
        "dataset_name": hf_ds_path,
        "audio_column_name": "audio",
        "text_column_name": "transcription",
        "train_split_name": "train",
        "eval_split_name": "train",

        # Sample text for generation during eval
        "full_generation_sample_text": sample_text,

        # Audio filtering
        "max_duration_in_seconds": 15.0,
        "min_duration_in_seconds": 1.0,
        "max_tokens_length": 500,

        # Model — the converted checkpoint with discriminator
        "model_name_or_path": checkpoint_dir,
        "preprocessing_num_workers": 4,

        # Training
        "do_train": True,
        "num_train_epochs": num_epochs,
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": False,
        "per_device_train_batch_size": batch_size,
        "learning_rate": learning_rate,
        "adam_beta1": 0.8,
        "adam_beta2": 0.99,
        "warmup_steps": 10,
        "group_by_length": False,

        # Evaluation
        "do_eval": True,
        "eval_steps": eval_steps,
        "per_device_eval_batch_size": batch_size,
        "max_eval_samples": 25,
        "do_step_schedule_per_epoch": True,

        # Loss weights (from ylacombe's MMS configs)
        "weight_disc": 3,
        "weight_fmaps": 1,
        "weight_gen": 1,
        "weight_kl": 1.5,
        "weight_duration": 1,
        "weight_mel": 35,

        # Precision and reproducibility
        "fp16": True,
        "seed": 456,

        # Checkpointing
        "save_steps": eval_steps,
        "save_total_limit": 3,
        "logging_steps": 10,
    }

    config_path = os.path.join(output_dir, "training_config.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("  Training config written to {}".format(config_path))
    return config_path


def launch_training(repo_dir, config_path):
    """Launch the fine-tuning via accelerate."""
    train_script = os.path.join(repo_dir, "run_vits_finetuning.py")
    if not os.path.isfile(train_script):
        raise FileNotFoundError(
            "run_vits_finetuning.py not found at {}".format(train_script)
        )

    cmd = "accelerate launch {} {}".format(train_script, config_path)
    print("\n  Launching training:")
    print("  $ {}".format(cmd))
    print("  " + "-" * 50)

    # Run training in foreground so we see output
    process = subprocess.Popen(
        cmd, shell=True, cwd=repo_dir,
        stdout=sys.stdout, stderr=sys.stderr,
    )
    return_code = process.wait()

    if return_code != 0:
        print("\n  Training exited with code {}".format(return_code))
        return False
    return True


def export_final_model(output_dir, export_dir):
    """Copy the final model to the export directory for easy inference."""
    # The training script saves the model in output_dir directly
    required_files = ["config.json"]
    model_files = [f for f in os.listdir(output_dir) if f.endswith((".bin", ".safetensors", ".json"))]

    if not model_files:
        print("  WARNING: No model files found in {}".format(output_dir))
        return False

    os.makedirs(export_dir, exist_ok=True)
    for f in os.listdir(output_dir):
        if f.endswith((".bin", ".safetensors", ".json", ".txt")):
            src = os.path.join(output_dir, f)
            dst = os.path.join(export_dir, f)
            if os.path.isfile(src):
                shutil.copy2(src, dst)

    print("  Model exported to {}".format(export_dir))
    return True


def test_inference(model_path, sample_text, output_wav):
    """Quick inference test with the fine-tuned model."""
    try:
        import torch
        import soundfile as sf
        from transformers import VitsModel, AutoTokenizer

        print("  Loading fine-tuned model...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = VitsModel.from_pretrained(model_path)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        inputs = tokenizer(sample_text, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(**inputs)

        waveform = output.waveform[0].cpu().numpy()
        sr = model.config.sampling_rate

        os.makedirs(os.path.dirname(output_wav), exist_ok=True)
        sf.write(output_wav, waveform, sr)

        duration = len(waveform) / sr
        print("  Generated {:.2f}s audio -> {}".format(duration, output_wav))
        print("  Text: '{}'".format(sample_text))
        return True
    except Exception as e:
        print("  Inference test failed: {}".format(e))
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune MMS-TTS-mlg on Malagasy speech data"
    )
    parser.add_argument(
        "--data-dir", default="/home/florent/milo/data/tts",
        help="Directory with prepared TTS data (from 06_prepare_tts_data.py)"
    )
    parser.add_argument(
        "--output-dir", default="/home/florent/milo/models/mms-tts-mg-ft",
        help="Output directory for fine-tuned model"
    )
    parser.add_argument(
        "--repo-dir", default="/home/florent/milo/tools/finetune-hf-vits",
        help="Path to clone ylacombe/finetune-hf-vits"
    )
    parser.add_argument(
        "--base-model", default="facebook/mms-tts-mlg",
        help="Base model to fine-tune"
    )
    parser.add_argument(
        "--lang-code", default="mlg",
        help="ISO 639-3 language code (default: mlg for Malagasy)"
    )
    parser.add_argument(
        "--epochs", type=int, default=200,
        help="Number of training epochs (default: 200)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Per-device batch size (default: 16, reduce if OOM)"
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5,
        help="Learning rate (default: 2e-5)"
    )
    parser.add_argument(
        "--eval-steps", type=int, default=50,
        help="Evaluate and save every N steps (default: 50)"
    )
    parser.add_argument(
        "--target-sr", type=int, default=16000,
        help="Target sample rate (default: 16000, must match MMS-TTS)"
    )
    parser.add_argument(
        "--sample-text", default="Manao ahoana tompoko, faly mifankahita aminao aho",
        help="Sample text for generation during eval"
    )
    parser.add_argument(
        "--skip-setup", action="store_true",
        help="Skip repo cloning and dependency installation"
    )
    parser.add_argument(
        "--skip-convert", action="store_true",
        help="Skip discriminator conversion (use existing)"
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip training (only do setup and config generation)"
    )
    parser.add_argument(
        "--test-only", action="store_true",
        help="Only run inference test on existing model"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("MILO -- Fine-tune MMS-TTS for Malagasy")
    print("=" * 60)
    print("Base model:  {}".format(args.base_model))
    print("Data dir:    {}".format(args.data_dir))
    print("Output dir:  {}".format(args.output_dir))
    print("Repo dir:    {}".format(args.repo_dir))
    print("Epochs:      {}".format(args.epochs))
    print("Batch size:  {}".format(args.batch_size))
    print("LR:          {}".format(args.lr))
    print("Eval steps:  {}".format(args.eval_steps))
    print("=" * 60)

    # Quick inference test mode
    if args.test_only:
        print("\n[TEST] Running inference on fine-tuned model...")
        test_wav = os.path.join(args.output_dir, "test_output.wav")
        test_inference(args.output_dir, args.sample_text, test_wav)
        return

    # Verify data directory
    metadata_path = os.path.join(args.data_dir, "metadata.csv")
    if not os.path.isfile(metadata_path):
        print("\nERROR: metadata.csv not found at {}".format(metadata_path))
        print("Run 06_prepare_tts_data.py first to prepare the training data.")
        sys.exit(1)

    # Count samples
    with open(metadata_path, "r", encoding="utf-8") as f:
        num_samples = sum(1 for _ in f) - 1  # minus header
    print("\nTraining data: {} samples".format(num_samples))

    # ----------------------------------------------------------------
    # Step 1: Setup finetune-hf-vits repo
    # ----------------------------------------------------------------
    if not args.skip_setup:
        print("\n[1/6] Setting up finetune-hf-vits repository...")
        ensure_repo_cloned(args.repo_dir)
        ensure_requirements(args.repo_dir)
        build_monotonic_align(args.repo_dir)
    else:
        print("\n[1/6] Skipping setup (--skip-setup)")

    # ----------------------------------------------------------------
    # Step 2: Convert discriminator
    # ----------------------------------------------------------------
    if not args.skip_convert:
        print("\n[2/6] Converting discriminator checkpoint...")
        checkpoint_dir = convert_discriminator(
            args.repo_dir, args.lang_code, args.output_dir
        )
    else:
        print("\n[2/6] Skipping conversion (--skip-convert)")
        checkpoint_dir = os.path.join(args.output_dir, "converted_checkpoint")
        if not os.path.isdir(checkpoint_dir):
            print("  ERROR: Converted checkpoint not found at {}".format(checkpoint_dir))
            print("  Run without --skip-convert first.")
            sys.exit(1)

    # ----------------------------------------------------------------
    # Step 3: Check tokenizer coverage
    # ----------------------------------------------------------------
    print("\n[3/6] Checking tokenizer coverage...")
    missing_chars = check_tokenizer_coverage(checkpoint_dir, args.data_dir)

    # ----------------------------------------------------------------
    # Step 4: Prepare HuggingFace dataset
    # ----------------------------------------------------------------
    print("\n[4/6] Preparing HuggingFace dataset...")
    hf_ds_path = prepare_hf_dataset(args.data_dir, args.target_sr)

    # ----------------------------------------------------------------
    # Step 5: Generate training config
    # ----------------------------------------------------------------
    print("\n[5/6] Generating training configuration...")
    training_output_dir = os.path.join(args.output_dir, "training_output")
    config_path = generate_training_config(
        checkpoint_dir=checkpoint_dir,
        data_dir=args.data_dir,
        output_dir=training_output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_steps=args.eval_steps,
        sample_text=args.sample_text,
        target_sr=args.target_sr,
    )

    # ----------------------------------------------------------------
    # Step 6: Launch training
    # ----------------------------------------------------------------
    if not args.skip_train:
        print("\n[6/6] Launching training...")
        print("  Config: {}".format(config_path))
        print("  This will take ~20-30 minutes on RTX 5070 Ti...")
        print("  Monitor with: tensorboard --logdir {}".format(training_output_dir))

        success = launch_training(args.repo_dir, config_path)

        if success:
            print("\n" + "=" * 60)
            print("TRAINING COMPLETE")
            print("=" * 60)

            # Export to clean directory
            export_dir = os.path.join(args.output_dir, "final")
            print("\nExporting final model to {}...".format(export_dir))
            export_final_model(training_output_dir, export_dir)

            # Quick inference test
            print("\nRunning inference test...")
            test_wav = os.path.join(args.output_dir, "test_output.wav")
            test_inference(export_dir, args.sample_text, test_wav)

            print("\n" + "=" * 60)
            print("DONE")
            print("=" * 60)
            print("  Fine-tuned model: {}".format(export_dir))
            print("  Test audio:       {}".format(test_wav))
            print("  TensorBoard:      tensorboard --logdir {}".format(training_output_dir))
            print("\n  To use in code:")
            print('    from transformers import VitsModel, AutoTokenizer')
            print('    model = VitsModel.from_pretrained("{}")'.format(export_dir))
            print('    tokenizer = AutoTokenizer.from_pretrained("{}")'.format(export_dir))
        else:
            print("\n  Training failed. Check logs above for errors.")
            print("  Common fixes:")
            print("    - Reduce --batch-size (try 8 or 4) if OOM")
            print("    - Check that monotonic_align built correctly")
            print("    - Ensure transformers>=4.35.1 is installed")
            sys.exit(1)
    else:
        print("\n[6/6] Skipping training (--skip-train)")
        print("  Config saved to: {}".format(config_path))
        print("\n  To launch manually:")
        print("  cd {} && accelerate launch run_vits_finetuning.py {}".format(
            args.repo_dir, config_path
        ))


if __name__ == "__main__":
    main()
