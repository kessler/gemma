import { Gemma } from '../src/index.js'

const gemma = new Gemma()
await gemma.load()

// One-shot
const answer = await gemma.complete('What is the speed of light?')
console.log(answer)

// Streaming
for await (const chunk of gemma.stream('Write a haiku about TypeScript')) {
  process.stdout.write(chunk)
}
console.log()

await gemma.unload()
