import { Gemma } from '../src/index.js'

const gemma = new Gemma()
await gemma.load()

const raw = await gemma.generateRaw(
  `<|turn>system\nYou are a helpful assistant.<|tool>declaration:read_file{"description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File path"}},"required":["path"]}}<tool|><turn|>\n<|turn>user\nRead the file package.json<turn|>\n<|turn>model\n`,
  { maxTokens: 200 }
)
console.log('RAW:', JSON.stringify(raw))

await gemma.unload()
