import { TextStreamer } from '@huggingface/transformers'

const SPECIAL_TOKENS = new Set([
  '<eos>', '<bos>', '<end_of_turn>', '<start_of_turn>',
  '<|turn>', '<turn|>',
  '<|tool>', '<tool|>',
  '<|tool_call>', '<tool_call|>',
  '<|tool_response>', '<tool_response|>',
  '<|channel>', '<channel|>',
  '<|think|>', '<|image|>',
  '<|"|>',
])

export function stripSpecialTokens(text: string): string {
  let result = text
  for (const token of SPECIAL_TOKENS) {
    if (result.includes(token)) {
      result = result.split(token).join('')
    }
  }
  return result
}

export interface StreamCallbacks {
  onChunk?: (text: string) => void
  onThinkingChunk?: (text: string) => void
}

export interface CreateStreamerResult {
  streamer: InstanceType<typeof TextStreamer>
  getRawResult: () => string
}

/**
 * Creates a TextStreamer that filters special tokens and separates thinking blocks.
 * Returns the streamer and a function to get the raw (unfiltered) result.
 */
export function createStreamer(
  tokenizer: any,
  callbacks: StreamCallbacks,
): CreateStreamerResult {
  let rawResult = ''
  let insideThinking = false
  let insideToolCall = false

  const streamer = new TextStreamer(tokenizer, {
    skip_prompt: true,
    skip_special_tokens: false,
    callback_function: (text: string) => {
      rawResult += text

      // Track thinking blocks
      if (text.includes('<|channel>')) {
        insideThinking = true
        return
      }
      if (text.includes('<channel|>')) {
        insideThinking = false
        return
      }
      if (insideThinking) {
        const clean = text.replace(/^thought\n?/, '')
        if (clean) callbacks.onThinkingChunk?.(clean)
        return
      }

      // Track tool call blocks
      if (text.includes('<|tool_call>')) insideToolCall = true
      if (text.includes('<tool_call|>') || text.includes('<tool_response|>')) {
        insideToolCall = false
        return
      }
      if (insideToolCall || text.includes('<|tool_response>')) return

      const clean = stripSpecialTokens(text)
      if (clean) callbacks.onChunk?.(clean)
    },
  })

  return { streamer, getRawResult: () => rawResult }
}

/**
 * Wraps a generate call into an AsyncIterableIterator.
 * The generator function receives an onChunk callback and should call it for each chunk.
 */
export async function* streamToAsyncIterator(
  generateFn: (onChunk: (text: string) => void) => Promise<void>,
): AsyncGenerator<string, void, undefined> {
  const chunks: string[] = []
  let resolve: (() => void) | null = null
  let done = false
  let error: Error | null = null

  const onChunk = (text: string) => {
    chunks.push(text)
    resolve?.()
  }

  const generatePromise = generateFn(onChunk).then(
    () => { done = true; resolve?.() },
    (err: Error) => { error = err; done = true; resolve?.() },
  )

  while (true) {
    if (chunks.length > 0) {
      yield chunks.shift()!
      continue
    }

    if (done) {
      if (error) throw error
      break
    }

    await new Promise<void>(r => { resolve = r })
    resolve = null
  }

  await generatePromise
}
