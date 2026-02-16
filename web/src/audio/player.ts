/**
 * Audio player: decodes base64 WAV and plays via Web Audio API.
 * Queues audio to avoid overlapping playback.
 */
export class AudioPlayer {
  private context: AudioContext | null = null;
  private queue: ArrayBuffer[] = [];
  private playing = false;

  private getContext(): AudioContext {
    if (!this.context) {
      this.context = new AudioContext();
    }
    return this.context;
  }

  async playBase64(b64: string): Promise<void> {
    const binary = atob(b64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    this.queue.push(bytes.buffer);
    await this.processQueue();
  }

  async playBlob(blob: Blob): Promise<void> {
    const buffer = await blob.arrayBuffer();
    this.queue.push(buffer);
    await this.processQueue();
  }

  private async processQueue(): Promise<void> {
    if (this.playing) return;
    this.playing = true;

    while (this.queue.length > 0) {
      const buffer = this.queue.shift()!;
      const ctx = this.getContext();

      try {
        const audioBuffer = await ctx.decodeAudioData(buffer.slice(0));
        await new Promise<void>((resolve) => {
          const source = ctx.createBufferSource();
          source.buffer = audioBuffer;
          source.connect(ctx.destination);
          source.onended = () => resolve();
          source.start();
        });
      } catch (err) {
        console.error("Audio playback error:", err);
      }
    }

    this.playing = false;
  }

  stop(): void {
    this.queue = [];
    this.playing = false;
  }
}
