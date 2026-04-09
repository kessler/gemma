import { Gemma } from '../src/index.js'

const gemma = new Gemma()
await gemma.load()

const response = await gemma.complete([{
  role: 'user',
  content: [
    { type: 'image', image: process.argv[2] ?? './photo.jpg' },
    { type: 'text', text: 'What do you see in this image?' },
  ],
}])

console.log(response)
await gemma.unload()
