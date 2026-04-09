import { Gemma } from '../src/index.js'

const gemma = new Gemma()
await gemma.load()

const response = await gemma.complete('What is 137 * 29? Show your work.', {
  thinking: true,
  onThinkingChunk: (t) => process.stderr.write(t),
})

console.log(response)
await gemma.unload()
