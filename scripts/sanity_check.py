"""Quick sanity check: verify the full training pipeline works."""
import torch
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration

print("=== Pipeline sanity check ===")

processor = WhisperProcessor.from_pretrained("openai/whisper-medium")

ds = load_dataset("badrex/malagasy-speech-full", split="train", streaming=True)
samples = []
for i, s in enumerate(ds):
    if i >= 3:
        break
    samples.append(s)

for i, s in enumerate(samples):
    audio = s["audio"]
    features = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
    )
    labels = processor.tokenizer(s["transcription"]).input_ids
    text_preview = s["transcription"][:60]
    print(
        "[{}] features: {} | labels: {} | {}...".format(
            i, features.input_features.shape, len(labels), text_preview
        )
    )

print("\nLoading model on GPU...")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
model = model.to("cuda").half()

audio = samples[0]["audio"]
input_features = processor.feature_extractor(
    audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
).input_features.to("cuda").half()
labels = processor.tokenizer(
    samples[0]["transcription"], return_tensors="pt"
).input_ids.to("cuda")

with torch.no_grad():
    outputs = model(input_features=input_features, labels=labels)
    loss = outputs.loss.item()
    vram = torch.cuda.memory_allocated() / 1e9
    print("Loss: {:.4f}".format(loss))
    print("VRAM used: {:.1f} GB".format(vram))

print("\n=== Pipeline OK ===")
