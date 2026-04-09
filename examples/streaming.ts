import { Gemma } from '../src/index.js'

const gemma = new Gemma()
await gemma.load()

for await (const chunk of gemma.stream('Explain quantum entanglement')) {
  process.stdout.write(chunk)
}
console.log()

await gemma.unload()
