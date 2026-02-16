/**
 * Audio recorder: captures 16kHz mono PCM in 320ms chunks.
 */
export class AudioRecorder {
  private context: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private processor: ScriptProcessorNode | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private onChunk: ((pcm: ArrayBuffer) => void) | null = null;
  private chunkCount = 0;

  private readonly targetSampleRate = 16000;
  private readonly chunkDurationMs = 320;

  async start(onChunk: (pcm: ArrayBuffer) => void): Promise<void> {
    this.onChunk = onChunk;
    this.chunkCount = 0;

    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: { ideal: this.targetSampleRate },
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
      },
    });

    this.context = new AudioContext({ sampleRate: this.targetSampleRate });
    console.log("[Recorder] AudioContext sampleRate:", this.context.sampleRate);
    this.source = this.context.createMediaStreamSource(this.stream);

    // Buffer size for ~320ms at 16kHz = 5120 samples
    const bufferSize = 4096;
    this.processor = this.context.createScriptProcessor(bufferSize, 1, 1);

    let buffer = new Float32Array(0);
    const chunkSamples = Math.floor(
      this.targetSampleRate * (this.chunkDurationMs / 1000)
    );

    this.processor.onaudioprocess = (e) => {
      const input = e.inputBuffer.getChannelData(0);
      const merged = new Float32Array(buffer.length + input.length);
      merged.set(buffer);
      merged.set(input, buffer.length);
      buffer = merged;

      while (buffer.length >= chunkSamples) {
        const chunk = buffer.slice(0, chunkSamples);
        buffer = buffer.slice(chunkSamples);

        // Convert float32 to int16 PCM
        const pcm = new Int16Array(chunk.length);
        for (let i = 0; i < chunk.length; i++) {
          const s = Math.max(-1, Math.min(1, chunk[i]!));
          pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }

        this.chunkCount++;
        if (this.chunkCount <= 3 || this.chunkCount % 30 === 0) {
          console.log("[Recorder] Chunk #%d: %d bytes", this.chunkCount, pcm.buffer.byteLength);
        }
        this.onChunk?.(pcm.buffer);
      }
    };

    this.source.connect(this.processor);
    this.processor.connect(this.context.destination);
    console.log("[Recorder] Started, chunkSamples=%d", chunkSamples);
  }

  stop(): void {
    console.log("[Recorder] Stopped after %d chunks", this.chunkCount);
    this.processor?.disconnect();
    this.source?.disconnect();
    this.stream?.getTracks().forEach((t) => t.stop());
    void this.context?.close();

    this.processor = null;
    this.source = null;
    this.stream = null;
    this.context = null;
    this.onChunk = null;
  }

  get isRecording(): boolean {
    return this.context !== null && this.context.state === "running";
  }
}
