import {
  Gemma4ForConditionalGeneration,
  AutoProcessor,
  TextStreamer,
  load_image,
} from '@huggingface/transformers'
import { readFile } from 'fs/promises'
import decode from 'audio-decode'
import type {
  GemmaOptions,
  CompleteOptions,
  CompletionInput,
  ChatMessage,
  ContentItem,
  ProgressInfo,
  DeviceType,
} from './types.js'
import { DEFAULT_MODEL_ID, resolveModelId } from './models.js'
import { createStreamer, streamToAsyncIterator } from './streaming.js'
import { extractFinalResponse } from '@kessler/gemma-agent'
import type { MediaAttachment } from '@kessler/gemma-agent'

export class Gemma {
  private model: InstanceType<typeof Gemma4ForConditionalGeneration> | null = null
  private processor: Awaited<ReturnType<typeof AutoProcessor.from_pretrained>> | null = null
  private loading = false
  private abortController: AbortController | null = null

  private readonly modelId: string
  private readonly device: DeviceType
  private readonly dtype: string
  private readonly onProgress?: (info: ProgressInfo) => void

  contextLimit = 128_000

  constructor(options: GemmaOptions = {}) {
    this.modelId = options.model ?? DEFAULT_MODEL_ID
    this.device = options.device ?? 'gpu'
    this.dtype = options.dtype ?? 'q4f16'
    this.onProgress = options.onProgress
  }

  async load(): Promise<void> {
    if (this.model) return
    if (this.loading) return
    this.loading = true

    const { hfModelId, contextLimit } = resolveModelId(this.modelId)
    this.contextLimit = contextLimit

    const fileProgress = new Map<string, { loaded: number; total: number }>()
    let lastReportedProgress = -1

    const progress_callback = (info: { status: string; file?: string; progress?: number; loaded?: number; total?: number }) => {
      if (info.status === 'progress' && info.file != null) {
        fileProgress.set(info.file, { loaded: info.loaded ?? 0, total: info.total ?? 0 })
        let totalBytes = 0
        let loadedBytes = 0
        for (const entry of fileProgress.values()) {
          totalBytes += entry.total
          loadedBytes += entry.loaded
        }
        const overall = totalBytes > 0 ? Math.round((loadedBytes / totalBytes) * 100) : 0
        if (overall !== lastReportedProgress) {
          lastReportedProgress = overall
          this.onProgress?.({ status: 'loading', progress: overall, file: info.file })
        }
      } else if (info.status === 'ready') {
        this.onProgress?.({ status: 'ready' })
      }
    }

    try {
      const [model, processor] = await Promise.all([
        Gemma4ForConditionalGeneration.from_pretrained(hfModelId, {
          dtype: this.dtype as any,
          device: this.device as any,
          progress_callback,
        }),
        AutoProcessor.from_pretrained(hfModelId),
      ])

      this.model = model as InstanceType<typeof Gemma4ForConditionalGeneration>
      this.processor = processor
      this.loading = false
      this.onProgress?.({ status: 'ready' })
    } catch (e) {
      this.loading = false
      this.onProgress?.({ status: 'error', error: String(e) })
      throw e
    }
  }

  isLoaded(): boolean {
    return this.model !== null
  }

  async unload(): Promise<void> {
    if (this.model) {
      await this.model.dispose()
      this.model = null
    }
    this.processor = null
    this.loading = false
  }

  abort(): void {
    if (this.abortController) {
      this.abortController.abort()
      this.abortController = null
    }
  }

  /**
   * Run a completion. Accepts a string (single user message) or a ChatMessage array.
   */
  async complete(input: CompletionInput, options?: CompleteOptions): Promise<string> {
    const { prompt, images, audios } = await this.prepareInput(input, options)
    const raw = await this.generate(prompt, images, audios, options)
    return extractFinalResponse(raw)
  }

  /**
   * Stream a completion. Returns an async iterator of text chunks.
   */
  async *stream(input: CompletionInput, options?: CompleteOptions): AsyncGenerator<string, void, undefined> {
    const { prompt, images, audios } = await this.prepareInput(input, options)

    yield* streamToAsyncIterator(async (onChunk) => {
      await this.generate(prompt, images, audios, {
        ...options,
        onChunk,
      })
    })
  }

  countTokens(text: string): number {
    if (!this.processor) {
      throw new Error('Cannot count tokens: model not loaded')
    }
    const { input_ids } = (this.processor as any).tokenizer(text, { add_special_tokens: false })
    return input_ids.size
  }

  /**
   * Low-level generate for agent use. Accepts a raw prompt string with Gemma 4 special tokens.
   * Used by the Agent class for tool-calling prompts.
   */
  async generateRaw(prompt: string, options?: CompleteOptions & { media?: MediaAttachment[] }): Promise<string> {
    const images: string[] = []
    const audios: string[] = []
    if (options?.media) {
      for (const m of options.media) {
        if (m.type === 'image') images.push(m.content)
        if (m.type === 'audio') audios.push(m.content)
      }
    }
    return this.generate(prompt, images, audios, options)
  }

  // ---- Private ----

  private async prepareInput(
    input: CompletionInput,
    options?: CompleteOptions,
  ): Promise<{ prompt: string; images: (string | any)[]; audios: (string | any)[] }> {
    if (!this.processor) throw new Error('Model not loaded. Call load() first.')

    const messages = normalizeInput(input)
    const images: any[] = []
    const audios: any[] = []

    // Collect media from content arrays
    for (const msg of messages) {
      if (typeof msg.content === 'string') continue
      for (const item of msg.content) {
        if (item.type === 'image') {
          images.push(item.image)
        } else if (item.type === 'audio') {
          audios.push(item.audio)
        }
      }
    }

    // Build messages for apply_chat_template (with placeholder content items)
    const templateMessages = messages.map(msg => {
      if (typeof msg.content === 'string') {
        return { role: msg.role, content: msg.content }
      }
      // Build content array for template
      const content: any[] = msg.content.map(item => {
        if (item.type === 'text') return { type: 'text', text: item.text }
        if (item.type === 'image') return { type: 'image' }
        if (item.type === 'audio') return { type: 'audio' }
        return item
      })
      return { role: msg.role, content }
    })

    const prompt = (this.processor as any).apply_chat_template(templateMessages, {
      tokenize: false,
      add_generation_prompt: true,
      enable_thinking: options?.thinking ?? false,
    })

    return { prompt, images, audios }
  }

  private async generate(
    prompt: string,
    images: any[],
    audios: any[],
    options?: CompleteOptions,
  ): Promise<string> {
    if (!this.model || !this.processor) throw new Error('Model not loaded. Call load() first.')

    // Load media
    const loadedImages = await Promise.all(
      images.map(img => {
        if (typeof img === 'string') return load_image(img)
        return img // assume already loaded (Buffer or RawImage)
      }),
    )

    const loadedAudios = await Promise.all(
      audios.map(audio => loadAudio(audio)),
    )

    // Tokenize
    let inputs: any
    const hasImages = loadedImages.length > 0
    const hasAudios = loadedAudios.length > 0

    if (hasImages || hasAudios) {
      inputs = await (this.processor as any)(
        prompt,
        hasImages ? (loadedImages.length === 1 ? loadedImages[0] : loadedImages) : null,
        hasAudios ? (loadedAudios.length === 1 ? loadedAudios[0] : loadedAudios) : null,
        { add_special_tokens: false },
      )
    } else {
      inputs = (this.processor as any).tokenizer(prompt, {
        add_special_tokens: false,
        return_tensor: 'pt',
      })
    }

    // Create streamer
    const { streamer, getRawResult } = createStreamer(
      (this.processor as any).tokenizer,
      {
        onChunk: options?.onChunk,
        onThinkingChunk: options?.onThinkingChunk,
      },
    )

    // Generate
    this.abortController = new AbortController()
    try {
      await this.model.generate({
        ...inputs,
        max_new_tokens: options?.maxTokens ?? 1024,
        do_sample: false,
        streamer,
        abort_signal: this.abortController.signal,
      })
    } catch (e) {
      if (e instanceof DOMException && e.name === 'AbortError') {
        return getRawResult()
      }
      throw e
    } finally {
      this.abortController = null
    }

    return getRawResult()
  }
}

function normalizeInput(input: CompletionInput): ChatMessage[] {
  if (typeof input === 'string') {
    return [{ role: 'user', content: input }]
  }
  return input
}

async function loadAudio(audio: string | Buffer | Float32Array): Promise<Float32Array> {
  if (audio instanceof Float32Array) return audio

  let buffer: ArrayBuffer
  if (typeof audio === 'string') {
    if (audio.startsWith('http://') || audio.startsWith('https://')) {
      const response = await fetch(audio)
      buffer = await response.arrayBuffer()
    } else {
      const fileBuffer = await readFile(audio)
      buffer = fileBuffer.buffer.slice(fileBuffer.byteOffset, fileBuffer.byteOffset + fileBuffer.byteLength) as ArrayBuffer
    }
  } else {
    buffer = audio.buffer.slice(audio.byteOffset, audio.byteOffset + audio.byteLength) as ArrayBuffer
  }

  const { channelData, sampleRate: _ } = await decode(buffer)

  // Mix to mono if stereo
  if (channelData.length > 1) {
    const SCALING_FACTOR = Math.sqrt(2)
    const mono = channelData[0]
    for (let i = 0; i < mono.length; ++i) {
      mono[i] = (SCALING_FACTOR * (mono[i] + channelData[1][i])) / 2
    }
    return mono
  }

  return channelData[0]
}
