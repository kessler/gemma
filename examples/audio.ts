import { Gemma } from '../src/index.js'

const gemma = new Gemma()
await gemma.load()

const response = await gemma.complete([{
  role: 'user',
  content: [
    { type: 'audio', audio: process.argv[2] ?? './recording.wav' },
    { type: 'text', text: 'Transcribe this.' },
  ],
}])

console.log(response)
await gemma.unload()
