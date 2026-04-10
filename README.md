# @kessler/gemma

Run Google's Gemma 4 models entirely on-device, embedded in a Node.js process. Text, image, and audio in — text out. No API keys, no cloud, no network required after the initial model download.

Built on [`@huggingface/transformers`](https://github.com/huggingface/transformers.js) + ONNX Runtime. Ships with a built-in agent system for tool-calling workflows.

## Why

- **Fully local** — your data never leaves the machine
- **Multimodal** — text, image, and audio inputs through one unified API
- **128K context** — sizeable context window to work with
- **Agent system** — tool-calling loop with automatic retries and context management
- **Streaming** — async iterators for real-time output
- **Thinking mode** — chain-of-thought reasoning when you need it
- **Hardware accelerated** — GPU auto-detection (CoreML on Mac, CUDA on NVIDIA, DirectML on Windows)

## Install

```bash
npm install @kessler/gemma
# or
pnpm add @kessler/gemma
```

## Quick Start

> [examples/quick-start.ts](examples/quick-start.ts)

```ts
import { Gemma } from '@kessler/gemma'

const gemma = new Gemma()
await gemma.load()

// One-shot
const answer = await gemma.complete('What is the speed of light?')

// Streaming
for await (const chunk of gemma.stream('Write a haiku about TypeScript')) {
  process.stdout.write(chunk)
}

await gemma.unload()
```

## Models

Both models support text, image, and audio inputs with a 128K token context window.

| ID | Parameters | Download | HuggingFace |
|----|-----------|----------|-------------|
| `gemma-4-e2b` | 2.3B effective | ~500 MB | [`onnx-community/gemma-4-E2B-it-ONNX`](https://huggingface.co/onnx-community/gemma-4-E2B-it-ONNX) |
| `gemma-4-e4b` (default) | 4B effective | ~1.5 GB | [`onnx-community/gemma-4-E4B-it-ONNX`](https://huggingface.co/onnx-community/gemma-4-E4B-it-ONNX) |

Models are downloaded on first use and cached locally (`~/.cache/huggingface/`). You can also pass any ONNX-format HuggingFace model ID directly.

```ts
const gemma = new Gemma({ model: 'gemma-4-e4b' })
```

## Multimodal

The `complete()` and `stream()` methods accept either a plain string or a messages array. Multimodal content goes inline in the messages:

### Image

> [examples/image.ts](examples/image.ts)

```ts
const response = await gemma.complete([{
  role: 'user',
  content: [
    { type: 'image', image: './photo.jpg' },
    { type: 'text', text: 'What do you see in this image?' },
  ],
}])
```

Images can be a file path, URL, or `Buffer`.

### Audio

> [examples/audio.ts](examples/audio.ts)

```ts
const response = await gemma.complete([{
  role: 'user',
  content: [
    { type: 'audio', audio: './recording.wav' },
    { type: 'text', text: 'Transcribe this.' },
  ],
}])
```

Audio can be a file path, URL, or `Buffer`. Max 30 seconds.

## Chat

> [examples/chat.ts](examples/chat.ts)

Multi-turn conversations use the standard `system` / `user` / `assistant` roles:

```ts
const response = await gemma.complete([
  { role: 'system', content: 'You are a concise technical writer.' },
  { role: 'user', content: 'Explain garbage collection in one paragraph.' },
])
```

## Streaming

> [examples/streaming.ts](examples/streaming.ts)

```ts
for await (const chunk of gemma.stream('Explain quantum entanglement')) {
  process.stdout.write(chunk)
}
```

Both `complete()` and `stream()` accept the same inputs — strings, message arrays, multimodal content.

## Thinking Mode

> [examples/thinking.ts](examples/thinking.ts)

Enable chain-of-thought reasoning. The model will reason internally before responding:

```ts
const response = await gemma.complete('What is 137 * 29? Show your work.', {
  thinking: true,
  onThinkingChunk: (t) => process.stderr.write(t),
})
```

## Device Selection

By default, `device: 'gpu'` auto-selects the best available backend:

| Platform | Backend |
|----------|---------|
| macOS (Apple Silicon) | CoreML / Metal |
| Linux / Windows (NVIDIA) | CUDA |
| Windows (any GPU) | DirectML |
| Fallback | CPU |

Override explicitly if needed:

```ts
const gemma = new Gemma({ device: 'cpu' })
const gemma = new Gemma({ device: 'cuda' })
const gemma = new Gemma({ device: 'coreml' })
```

## Download Progress

> [examples/progress.ts](examples/progress.ts)

Track model download and loading:

```ts
const gemma = new Gemma({
  onProgress: (info) => {
    if (info.status === 'loading') console.log(`${info.progress}%`)
    if (info.status === 'ready') console.log('Model ready')
    if (info.status === 'error') console.error(info.error)
  },
})
```

## Agent

> [examples/agent.ts](examples/agent.ts)

The agent runs an autonomous tool-calling loop: the model decides which tools to call, executes them, reads the results, and continues until it has an answer.

```ts
import { Gemma, Agent } from '@kessler/gemma'
import fs from 'fs/promises'

const gemma = new Gemma({ model: 'gemma-4-e4b' })
await gemma.load()

const agent = new Agent({
  gemma,
  systemPrompt: 'You are a helpful file assistant.',
  tools: [
    {
      name: 'read_file',
      description: 'Read a file from disk',
      parameters: {
        type: 'object',
        properties: {
          path: { type: 'string', description: 'File path to read' },
        },
        required: ['path'],
      },
      execute: async (args) => {
        return { content: await fs.readFile(args.path as string, 'utf-8') }
      },
    },
  ],
  onChunk: (text) => process.stdout.write(text),
  onToolCall: (call) => console.log(`\n> ${call.name}(${JSON.stringify(call.arguments)})`),
})

const result = await agent.run('Read package.json and tell me the project name')
console.log(`\nDone in ${result.iterations} iterations, ${result.toolCallCount} tool calls`)
```

### Agent Features

- **Self-executing tools** — each tool definition carries its own `execute` function, no separate executor needed
- **Persistent conversation** — call `agent.run()` multiple times, context carries over
- **Truncation recovery** — if a tool call gets cut off mid-generation, the agent automatically continues or compresses context
- **Image handling** — tool results containing screenshots are fed back through the multimodal processor
- **Abort support** — call `agent.abort()` to stop mid-generation

```ts
// Multi-turn
const r1 = await agent.run('List the files in src/')
const r2 = await agent.run('Now read the main entry point')

// Reset
agent.clearHistory()
```

## API Reference

### `new Gemma(options?)`

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `model` | `string` | `'gemma-4-e4b'` | Model ID or HuggingFace model path |
| `device` | `'gpu' \| 'cpu' \| 'cuda' \| 'coreml'` | `'gpu'` | Inference device |
| `dtype` | `string` | `'q4f16'` | Quantization type |
| `onProgress` | `(info: ProgressInfo) => void` | — | Download/load progress callback |

### `gemma.load(): Promise<void>`

Download (if needed) and load the model. Must be called before `complete()` or `stream()`.

### `gemma.complete(input, options?): Promise<string>`

Generate a completion. `input` is a `string` or `ChatMessage[]`.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `maxTokens` | `number` | `1024` | Maximum tokens to generate |
| `thinking` | `boolean` | `false` | Enable chain-of-thought reasoning |
| `onChunk` | `(text: string) => void` | — | Streaming text callback |
| `onThinkingChunk` | `(text: string) => void` | — | Streaming thinking callback |

### `gemma.stream(input, options?): AsyncGenerator<string>`

Same as `complete()` but yields text chunks as they're generated.

### `gemma.countTokens(text): number`

Returns the token count for a string.

### `gemma.unload(): Promise<void>`

Dispose the model and free memory.

### `tokenize(input): Token[]`

Low-level lexer for Gemma 4 model output. Splits a raw string into typed tokens (`TOOL_CALL_START`, `TEXT`, `STRING_DELIM`, etc.). Useful for building custom parsers on top of Gemma's special token format.

### `gemma.abort(): void`

Cancel an in-progress generation.

### `new Agent(options)`

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `gemma` | `Gemma` | — | Loaded Gemma instance (required) |
| `systemPrompt` | `string` | — | System prompt (required) |
| `tools` | `ToolDefinition[]` | — | Available tools (required) |
| `maxIterations` | `number` | `10` | Max tool-calling loops |
| `thinking` | `boolean` | `false` | Enable reasoning mode |
| `onChunk` | `(text: string) => void` | — | Streaming text callback |
| `onThinkingChunk` | `(text: string) => void` | — | Streaming thinking callback |
| `onToolCall` | `(call: ToolCall) => void` | — | Called when a tool is invoked |
| `onToolResponse` | `(resp: ToolResponse) => void` | — | Called when a tool returns |

### `agent.run(message): Promise<AgentRunResult>`

Run the agent. Returns `{ response, toolCallCount, iterations }`.

### `agent.clearHistory(): void`

Reset conversation state.

### `agent.abort(): void`

Stop the current run.

## Types

```ts
interface ChatMessage {
  role: 'system' | 'user' | 'assistant'
  content: string | ContentItem[]
}

type ContentItem =
  | { type: 'text'; text: string }
  | { type: 'image'; image: string | Buffer }
  | { type: 'audio'; audio: string | Buffer }

interface ToolDefinition {
  name: string
  description: string
  parameters?: { type: 'object'; properties: Record<string, ToolParameterDef>; required?: string[] }
  execute: (args: Record<string, unknown>) => Promise<Record<string, unknown>>
}

interface AgentRunResult {
  response: string
  toolCallCount: number
  iterations: number
}
```

## License

Apache-2.0
