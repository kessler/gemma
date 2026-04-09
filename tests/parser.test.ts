import { describe, it, expect } from 'vitest'
import { parseToolCalls, hasToolCalls, extractThinking, extractFinalResponse } from '../src/agent/parser.js'

// ─── parseToolCalls ───────────────────────────────────────────────

describe('parseToolCalls', () => {
  describe('Gemma custom format (STRING_DELIM args)', () => {
    it('parses a single tool call with string arg', () => {
      const text = '<|tool_call>call:read_file{path:<|"|>/tmp/foo.txt<|"|>}<tool_call|>'
      const calls = parseToolCalls(text)
      expect(calls).toEqual([
        { name: 'read_file', arguments: { path: '/tmp/foo.txt' } },
      ])
    })

    it('parses mixed string and numeric args', () => {
      const text = '<|tool_call>call:scroll_page{direction:<|"|>down<|"|>,amount:500}<tool_call|>'
      const calls = parseToolCalls(text)
      expect(calls).toEqual([
        { name: 'scroll_page', arguments: { direction: 'down', amount: 500 } },
      ])
    })

    it('parses boolean values', () => {
      const text = '<|tool_call>call:toggle{enabled:true}<tool_call|>'
      const calls = parseToolCalls(text)
      expect(calls[0].arguments.enabled).toBe(true)
    })

    it('parses false and null values', () => {
      const text = '<|tool_call>call:update{active:false,data:null}<tool_call|>'
      const calls = parseToolCalls(text)
      expect(calls[0].arguments.active).toBe(false)
      expect(calls[0].arguments.data).toBe(null)
    })

    it('parses float values', () => {
      const text = '<|tool_call>call:set_temp{value:0.7}<tool_call|>'
      const calls = parseToolCalls(text)
      expect(calls[0].arguments.value).toBe(0.7)
    })

    it('handles string value containing commas', () => {
      const text = '<|tool_call>call:write{content:<|"|>a,b,c<|"|>}<tool_call|>'
      const calls = parseToolCalls(text)
      expect(calls[0].arguments.content).toBe('a,b,c')
    })

    it('handles string value containing colons', () => {
      const text = '<|tool_call>call:open{url:<|"|>http://example.com<|"|>}<tool_call|>'
      const calls = parseToolCalls(text)
      expect(calls[0].arguments.url).toBe('http://example.com')
    })

    it('handles multiple string args', () => {
      const text = '<|tool_call>call:copy{src:<|"|>a.txt<|"|>,dest:<|"|>b.txt<|"|>}<tool_call|>'
      const calls = parseToolCalls(text)
      expect(calls[0].arguments).toEqual({ src: 'a.txt', dest: 'b.txt' })
    })
  })

  describe('JSON format args', () => {
    it('parses a single tool call with JSON args', () => {
      const text = '<|tool_call>call:read_file{"path":"package.json"}<tool_call|>'
      const calls = parseToolCalls(text)
      expect(calls).toEqual([
        { name: 'read_file', arguments: { path: 'package.json' } },
      ])
    })

    it('parses JSON args with multiple keys', () => {
      const text = '<|tool_call>call:search{"query":"hello","limit":10,"exact":true}<tool_call|>'
      const calls = parseToolCalls(text)
      expect(calls).toEqual([
        { name: 'search', arguments: { query: 'hello', limit: 10, exact: true } },
      ])
    })

    it('parses JSON args with nested objects', () => {
      const text = '<|tool_call>call:configure{"options":{"verbose":true,"timeout":30}}<tool_call|>'
      const calls = parseToolCalls(text)
      expect(calls[0].arguments).toEqual({
        options: { verbose: true, timeout: 30 },
      })
    })

    it('parses JSON args with arrays', () => {
      const text = '<|tool_call>call:batch{"ids":[1,2,3]}<tool_call|>'
      const calls = parseToolCalls(text)
      expect(calls[0].arguments).toEqual({ ids: [1, 2, 3] })
    })

    it('parses JSON args with null value', () => {
      const text = '<|tool_call>call:reset{"field":null}<tool_call|>'
      const calls = parseToolCalls(text)
      expect(calls[0].arguments.field).toBe(null)
    })
  })

  describe('empty and missing args', () => {
    it('parses empty args object', () => {
      const text = '<|tool_call>call:list_files{}<tool_call|>'
      const calls = parseToolCalls(text)
      expect(calls).toEqual([
        { name: 'list_files', arguments: {} },
      ])
    })
  })

  describe('multiple tool calls', () => {
    it('parses two consecutive tool calls', () => {
      const text =
        '<|tool_call>call:read_file{path:<|"|>a.txt<|"|>}<tool_call|>' +
        '<|tool_call>call:read_file{path:<|"|>b.txt<|"|>}<tool_call|>'
      const calls = parseToolCalls(text)
      expect(calls).toHaveLength(2)
      expect(calls[0].arguments.path).toBe('a.txt')
      expect(calls[1].arguments.path).toBe('b.txt')
    })

    it('parses tool calls with text between them', () => {
      const text =
        'Some preamble\n' +
        '<|tool_call>call:foo{"a":1}<tool_call|>' +
        '\nSome middle text\n' +
        '<|tool_call>call:bar{"b":2}<tool_call|>' +
        '\nSome trailing text'
      const calls = parseToolCalls(text)
      expect(calls).toHaveLength(2)
      expect(calls[0]).toEqual({ name: 'foo', arguments: { a: 1 } })
      expect(calls[1]).toEqual({ name: 'bar', arguments: { b: 2 } })
    })

    it('handles mixed JSON and Gemma format in same output', () => {
      const text =
        '<|tool_call>call:foo{"path":"a.txt"}<tool_call|>' +
        '<|tool_call>call:bar{path:<|"|>b.txt<|"|>}<tool_call|>'
      const calls = parseToolCalls(text)
      expect(calls).toHaveLength(2)
      expect(calls[0].arguments.path).toBe('a.txt')
      expect(calls[1].arguments.path).toBe('b.txt')
    })
  })

  describe('no tool calls', () => {
    it('returns empty array for plain text', () => {
      expect(parseToolCalls('Hello world')).toEqual([])
    })

    it('returns empty array for text with other special tokens', () => {
      expect(parseToolCalls('<|turn>model\nHello<turn|><eos>')).toEqual([])
    })
  })

  describe('edge cases', () => {
    it('handles truncated tool call (no TOOL_CALL_END)', () => {
      const text = '<|tool_call>call:read_file{"path":"pkg.json"}'
      const calls = parseToolCalls(text)
      // Should still parse the content before EOF
      expect(calls).toHaveLength(1)
      expect(calls[0].arguments.path).toBe('pkg.json')
    })

    it('handles tool call embedded in a full model turn', () => {
      const text =
        '<|turn>model\n' +
        '<|channel>thought\nI should read the file<channel|>' +
        '<|tool_call>call:read_file{"path":"package.json"}<tool_call|>' +
        '<turn|>'
      const calls = parseToolCalls(text)
      expect(calls).toHaveLength(1)
      expect(calls[0].name).toBe('read_file')
    })
  })
})

// ─── hasToolCalls ─────────────────────────────────────────────────

describe('hasToolCalls', () => {
  it('returns true when tool call token is present', () => {
    expect(hasToolCalls('text <|tool_call>call:foo{}<tool_call|>')).toBe(true)
  })

  it('returns false when no tool call token', () => {
    expect(hasToolCalls('just regular text')).toBe(false)
  })

  it('returns true even for incomplete tool call (start without end)', () => {
    expect(hasToolCalls('text <|tool_call>call:foo')).toBe(true)
  })
})

// ─── extractThinking ─────────────────────────────────────────────

describe('extractThinking', () => {
  it('extracts thinking block content', () => {
    const text = '<|channel>thought\nI should think about this\n<channel|>\nThe answer is 42.'
    const { thinking, rest } = extractThinking(text)
    expect(thinking).toBe('I should think about this')
    expect(rest).toContain('The answer is 42.')
  })

  it('returns empty thinking when no channel block', () => {
    const { thinking, rest } = extractThinking('No thinking here')
    expect(thinking).toBe('')
    expect(rest).toBe('No thinking here')
  })

  it('preserves special tokens in rest', () => {
    const text = '<|channel>thought\nhmm<channel|><|tool_call>call:foo{}<tool_call|>'
    const { rest } = extractThinking(text)
    expect(rest).toContain('<|tool_call>')
    expect(rest).toContain('<tool_call|>')
  })

  it('handles thinking without newline after "thought"', () => {
    const text = '<|channel>thoughtsome thinking<channel|>response'
    const { thinking } = extractThinking(text)
    expect(thinking).toBe('some thinking')
  })

  it('handles multiline thinking', () => {
    const text = '<|channel>thought\nLine 1\nLine 2\nLine 3<channel|>done'
    const { thinking } = extractThinking(text)
    expect(thinking).toBe('Line 1\nLine 2\nLine 3')
  })

  it('handles channel block that does not start with "thought"', () => {
    const text = '<|channel>some other content<channel|>response'
    const { thinking, rest } = extractThinking(text)
    expect(thinking).toBe('some other content')
    expect(rest).toContain('response')
  })

  it('removes thinking from rest completely', () => {
    const text = 'before<|channel>thought\nstuff<channel|>after'
    const { rest } = extractThinking(text)
    expect(rest).not.toContain('stuff')
    expect(rest).not.toContain('thought')
    expect(rest).toContain('before')
    expect(rest).toContain('after')
  })
})

// ─── extractFinalResponse ─────────────────────────────────────────

describe('extractFinalResponse', () => {
  it('strips thinking blocks', () => {
    const text = '<|turn>model\n<|channel>thought\nhmm<channel|>Hello world<eos>'
    expect(extractFinalResponse(text)).toBe('Hello world')
  })

  it('strips tool call and response blocks', () => {
    const text =
      '<|tool_call>call:foo{}<tool_call|>' +
      '<|tool_response>response:foo{ok:true}<tool_response|>' +
      'The result is good.'
    expect(extractFinalResponse(text)).toBe('The result is good.')
  })

  it('strips all standalone special tokens', () => {
    const text = '<bos><|turn>model\nHello<turn|><eos><end_of_turn>'
    expect(extractFinalResponse(text)).toBe('Hello')
  })

  it('strips image tokens', () => {
    const text = '<|turn>model\n<|image|>Description of image<eos>'
    expect(extractFinalResponse(text)).toBe('Description of image')
  })

  it('strips tool declaration blocks', () => {
    const text = '<|tool>declaration:foo{"description":"test"}<tool|>Response here'
    expect(extractFinalResponse(text)).toBe('Response here')
  })

  it('handles multiple turn markers', () => {
    const text = '<|turn>model\nFirst part<turn|>\n<|turn>model\nSecond part<turn|>'
    expect(extractFinalResponse(text)).toBe('First part\nSecond part')
  })

  it('strips the "model" role label after turn start', () => {
    const text = '<|turn>model\nActual response<turn|>'
    expect(extractFinalResponse(text)).toBe('Actual response')
  })

  it('strips the "system" role label after turn start', () => {
    const text = '<|turn>system\nYou are helpful<turn|>'
    expect(extractFinalResponse(text)).toBe('You are helpful')
  })

  it('returns empty string for content that is entirely special tokens', () => {
    const text = '<|tool_call>call:foo{}<tool_call|><|tool_response>resp<tool_response|>'
    expect(extractFinalResponse(text)).toBe('')
  })

  it('handles a realistic full model output', () => {
    const text =
      '<|turn>model\n' +
      '<|channel>thought\nLet me think about this...\nOk I know the answer.<channel|>' +
      '<|tool_call>call:read_file{"path":"pkg.json"}<tool_call|>' +
      '<|tool_response>response:read_file{content:<|"|>{"name":"test"}<|"|>}<tool_response|>' +
      'The project name is "test".' +
      '<turn|><eos>'
    expect(extractFinalResponse(text)).toBe('The project name is "test".')
  })

  it('preserves newlines in the response body', () => {
    const text = '<|turn>model\nLine 1\nLine 2\nLine 3<turn|><eos>'
    expect(extractFinalResponse(text)).toBe('Line 1\nLine 2\nLine 3')
  })

  it('handles think token', () => {
    const text = '<|think|><|turn>model\nHello<turn|>'
    expect(extractFinalResponse(text)).toBe('Hello')
  })
})
