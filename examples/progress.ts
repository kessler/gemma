import { Gemma } from '../src/index.js'

const gemma = new Gemma({
  onProgress: (info) => {
    if (info.status === 'loading') process.stdout.write(`\rDownloading... ${info.progress}%`)
    if (info.status === 'ready') console.log('\rModel ready.                ')
    if (info.status === 'error') console.error(info.error)
  },
})

await gemma.load()

const response = await gemma.complete('Hello!')
console.log(response)

await gemma.unload()
