import { describe, it, expect } from 'vitest'
import { stripSpecialTokens, streamToAsyncIterator } from '../src/streaming.js'

describe('stripSpecialTokens', () => {
  it('strips known special tokens', () => {
    expect(stripSpecialTokens('Hello<eos>')).toBe('Hello')
    expect(stripSpecialTokens('<|turn>model\nHi<turn|>')).toBe('model\nHi')
    expect(stripSpecialTokens('<bos>text<eos>')).toBe('text')
  })

  it('leaves regular text untouched', () => {
    expect(stripSpecialTokens('Hello world')).toBe('Hello world')
  })
})

describe('streamToAsyncIterator', () => {
  it('yields chunks from callback-based generator', async () => {
    const chunks: string[] = []

    const iter = streamToAsyncIterator(async (onChunk) => {
      onChunk('Hello')
      onChunk(' ')
      onChunk('world')
    })

    for await (const chunk of iter) {
      chunks.push(chunk)
    }

    expect(chunks.join('')).toBe('Hello world')
  })

  it('propagates errors', async () => {
    const iter = streamToAsyncIterator(async () => {
      throw new Error('boom')
    })

    await expect(async () => {
      for await (const _ of iter) { /* consume */ }
    }).rejects.toThrow('boom')
  })
})
