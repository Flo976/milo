"""Fine-tune Whisper medium on Malagasy speech data."""
import argparse
import os
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from datasets import load_dataset, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/whisper-medium")
    parser.add_argument("--dataset", default="badrex/malagasy-speech-full")
    parser.add_argument("--output-dir", default="/home/florent/milo/models/whisper-mg-v1")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-eval", type=int, default=None)
    parser.add_argument("--resume", default=None, help="Resume from checkpoint (path or 'auto' for latest)")
    args = parser.parse_args()

    print("=" * 60)
    print("MILO — Fine-tuning Whisper on Malagasy")
    print("=" * 60)
    print("GPU: {}".format(torch.cuda.get_device_name(0)))
    print("VRAM: {:.1f} GB".format(torch.cuda.get_device_properties(0).total_memory / 1e9))
    print("Model: {}".format(args.model))
    print("Dataset: {}".format(args.dataset))
    print("Output: {}".format(args.output_dir))
    print("=" * 60)

    # Load processor + model
    print("\n[1/4] Loading model and processor...")
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model)

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = None
    model.generation_config.language = "mg"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="mg", task="transcribe"
    )
    model.generation_config.suppress_tokens = []

    # Load dataset (from cache)
    print("\n[2/4] Loading dataset...")
    ds = load_dataset(args.dataset)

    train_ds = ds["train"]
    eval_ds = ds["validation"]

    if args.max_train:
        train_ds = train_ds.select(range(min(args.max_train, len(train_ds))))
    if args.max_eval:
        eval_ds = eval_ds.select(range(min(args.max_eval, len(eval_ds))))

    print("  Train: {} samples".format(len(train_ds)))
    print("  Eval:  {} samples".format(len(eval_ds)))

    # Resample to 16kHz
    train_ds = train_ds.cast_column("audio", Audio(sampling_rate=16000))
    eval_ds = eval_ds.cast_column("audio", Audio(sampling_rate=16000))

    # Preprocess
    print("\n[3/4] Preprocessing...")

    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
        return batch

    train_ds = train_ds.map(
        prepare_dataset,
        remove_columns=train_ds.column_names,
        num_proc=1,
    )
    eval_ds = eval_ds.map(
        prepare_dataset,
        remove_columns=eval_ds.column_names,
        num_proc=1,
    )

    # Metric
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # Training
    print("\n[4/4] Starting training...")
    effective_batch = args.batch_size * args.grad_accum
    print("  Epochs: {}".format(args.epochs))
    print("  Batch: {} x {} = {} effective".format(args.batch_size, args.grad_accum, effective_batch))
    print("  LR: {}".format(args.lr))
    print("  Eval every {} steps".format(args.eval_steps))

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        fp16=True,
        gradient_checkpointing=True,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to=["tensorboard"],
        dataloader_num_workers=4,
        remove_unused_columns=False,
        save_total_limit=3,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    resume_checkpoint = None
    if args.resume == "auto":
        resume_checkpoint = True  # Trainer finds latest automatically
        print("\nResuming from latest checkpoint...")
    elif args.resume:
        resume_checkpoint = args.resume
        print("\nResuming from {}...".format(resume_checkpoint))

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # Save final
    final_path = os.path.join(args.output_dir, "final")
    print("\nSaving final model to {}...".format(final_path))
    trainer.save_model(final_path)
    processor.save_pretrained(final_path)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
