import { Gemma } from '../src/index.js'

const gemma = new Gemma()
await gemma.load()

const response = await gemma.complete([
  { role: 'system', content: 'You are a concise technical writer.' },
  { role: 'user', content: 'Explain garbage collection in one paragraph.' },
])

console.log(response)
await gemma.unload()
