export type DeviceType = 'gpu' | 'cpu' | 'cuda' | 'coreml'

export interface GemmaOptions {
  /** Model ID: 'gemma-4-e2b', 'gemma-4-e4b', or a custom HuggingFace model ID */
  model?: string
  /** Device for inference. Default 'gpu' (auto-selects best backend per platform) */
  device?: DeviceType
  /** Quantization type. Default 'q4f16' */
  dtype?: string
  /** Progress callback during model download/load */
  onProgress?: (info: ProgressInfo) => void
}

export interface ProgressInfo {
  status: 'loading' | 'ready' | 'error'
  /** Overall progress percentage (0-100) */
  progress?: number
  /** Current file being downloaded */
  file?: string
  /** Error message if status is 'error' */
  error?: string
}

export type ContentItem =
  | { type: 'text'; text: string }
  | { type: 'image'; image: string | Buffer }
  | { type: 'audio'; audio: string | Buffer }

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant'
  content: string | ContentItem[]
}

export type CompletionInput = string | ChatMessage[]

export interface CompleteOptions {
  /** Max tokens to generate. Default 1024 */
  maxTokens?: number
  /** Enable thinking/reasoning mode */
  thinking?: boolean
  /** Callback for each streamed text chunk */
  onChunk?: (text: string) => void
  /** Callback for each thinking chunk */
  onThinkingChunk?: (text: string) => void
}

