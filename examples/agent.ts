import { Gemma, Agent } from '../src/index.js'
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
  thinking: true,
  onChunk: (text) => process.stdout.write(text),
  onToolCall: (call) => console.log(`\n> ${call.name}(${JSON.stringify(call.arguments)})`),
})

const result = await agent.run('Read the file at path "package.json" and tell me the project name')
console.log(`\nDone in ${result.iterations} iterations, ${result.toolCallCount} tool calls`)

await gemma.unload()
