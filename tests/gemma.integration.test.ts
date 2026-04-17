import { describe, it, expect, beforeAll, afterAll } from 'vitest'
import { Gemma, Agent } from '../src/index.js'
import fs from 'fs/promises'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const FIXTURES = path.join(__dirname, '..', 'examples', 'fixtures')

function defineModelTests(modelId: string) {
  describe(modelId, { timeout: 600_000 }, () => {
    let gemma: Gemma

    beforeAll(async () => {
      gemma = new Gemma({ model: modelId })
      await gemma.load()
    }, 600_000)

    afterAll(async () => {
      await gemma.unload()
    })

    // ─── Text ──────────────────────────────────────────────

    it('completes a text prompt', async () => {
      const response = await gemma.complete('What is 2 + 2? Answer with just the number.')
      expect(response).toContain('4')
    })

    it('streams a response', async () => {
      const chunks: string[] = []
      for await (const chunk of gemma.stream('Say hello.')) {
        chunks.push(chunk)
      }
      expect(chunks.length).toBeGreaterThan(0)
      expect(chunks.join('')).toBeTruthy()
    })

    it('counts tokens', () => {
      const count = gemma.countTokens('Hello world')
      expect(count).toBeGreaterThan(0)
      expect(count).toBeLessThan(10)
    })

    // ─── Chat ──────────────────────────────────────────────

    it('completes with chat messages (system + user roles)', async () => {
      const response = await gemma.complete([
        { role: 'system', content: 'You are a concise assistant. Reply in one sentence.' },
        { role: 'user', content: 'What is the capital of France?' },
      ])
      expect(response.toLowerCase()).toContain('paris')
    })

    // ─── Thinking ──────────────────────────────────────────

    it('completes with thinking mode', async () => {
      let thinkingOutput = ''
      const response = await gemma.complete('What is 15 * 13?', {
        thinking: true,
        onThinkingChunk: (t) => { thinkingOutput += t },
      })
      expect(response).toContain('195')
      expect(thinkingOutput.length).toBeGreaterThan(0)
    })

    // ─── Image ─────────────────────────────────────────────

    it('describes an image', async () => {
      const response = await gemma.complete([{
        role: 'user',
        content: [
          { type: 'image', image: path.join(FIXTURES, 'test.jpg') },
          { type: 'text', text: 'Describe what you see in this image in one sentence.' },
        ],
      }])
      expect(response.length).toBeGreaterThan(10)
    })

    // ─── Audio ─────────────────────────────────────────────

    it('transcribes audio', async () => {
      const response = await gemma.complete([{
        role: 'user',
        content: [
          { type: 'audio', audio: path.join(FIXTURES, 'test.wav') },
          { type: 'text', text: 'Transcribe this audio.' },
        ],
      }])
      expect(response.length).toBeGreaterThan(10)
    })

    // ─── Agent ─────────────────────────────────────────────

    it('runs agent with tool calls', async () => {
      const agent = new Agent({
        model: gemma,
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
      })

      const result = await agent.run('Read the file at path "package.json" and tell me the project name')
      expect(result.toolCallCount).toBeGreaterThan(0)
      expect(result.response.toLowerCase()).toContain('gemma')
    })
  })
}

describe('Gemma integration', () => {
  defineModelTests('gemma-4-e2b')
  defineModelTests('gemma-4-e4b')
})
