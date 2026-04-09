import { describe, it, expect } from 'vitest'
import { tokenize, type Token } from '../src/agent/lexer.js'

describe('tokenize', () => {
  it('returns a single TEXT token for plain text', () => {
    expect(tokenize('Hello world')).toEqual([
      { type: 'TEXT', value: 'Hello world' },
    ])
  })

  it('returns empty array for empty string', () => {
    expect(tokenize('')).toEqual([])
  })

  it('tokenizes a standalone special token', () => {
    expect(tokenize('<|tool_call>')).toEqual([
      { type: 'TOOL_CALL_START', value: '<|tool_call>' },
    ])
  })

  it('tokenizes text before and after a special token', () => {
    expect(tokenize('before<eos>after')).toEqual([
      { type: 'TEXT', value: 'before' },
      { type: 'EOS', value: '<eos>' },
      { type: 'TEXT', value: 'after' },
    ])
  })

  it('tokenizes consecutive special tokens with no text between', () => {
    expect(tokenize('<|tool_call><tool_call|>')).toEqual([
      { type: 'TOOL_CALL_START', value: '<|tool_call>' },
      { type: 'TOOL_CALL_END', value: '<tool_call|>' },
    ])
  })

  it('tokenizes all known special tokens', () => {
    const input = '<bos><eos><end_of_turn><|turn><turn|><|tool><tool|><|tool_call><tool_call|><|tool_response><tool_response|><|channel><channel|><|"|><|image|><|think|>'
    const types = tokenize(input).map(t => t.type)
    expect(types).toEqual([
      'BOS', 'EOS', 'END_OF_TURN',
      'TURN_START', 'TURN_END',
      'TOOL_DECL_START', 'TOOL_DECL_END',
      'TOOL_CALL_START', 'TOOL_CALL_END',
      'TOOL_RESPONSE_START', 'TOOL_RESPONSE_END',
      'CHANNEL_START', 'CHANNEL_END',
      'STRING_DELIM', 'IMAGE', 'THINK',
    ])
  })

  it('does not match partial special tokens', () => {
    // "<|tool" without the closing ">" is not a special token
    expect(tokenize('<|tool')).toEqual([
      { type: 'TEXT', value: '<|tool' },
    ])
  })

  it('does not match unknown angle-bracket sequences', () => {
    expect(tokenize('<div>hello</div>')).toEqual([
      { type: 'TEXT', value: '<div>hello</div>' },
    ])
  })

  it('handles a bare < that is not part of any token', () => {
    expect(tokenize('a < b')).toEqual([
      { type: 'TEXT', value: 'a < b' },
    ])
  })

  it('prefers longer token match (<|tool_call> over <|tool>)', () => {
    // The string starts with <|tool_call> — it should NOT be split into <|tool> + "call>"
    const tokens = tokenize('<|tool_call>data<tool_call|>')
    expect(tokens[0]).toEqual({ type: 'TOOL_CALL_START', value: '<|tool_call>' })
  })

  it('tokenizes a full Gemma-format tool call', () => {
    const input = '<|tool_call>call:read_file{path:<|"|>/tmp/foo.txt<|"|>}<tool_call|>'
    const tokens = tokenize(input)
    expect(tokens).toEqual([
      { type: 'TOOL_CALL_START', value: '<|tool_call>' },
      { type: 'TEXT', value: 'call:read_file{path:' },
      { type: 'STRING_DELIM', value: '<|"|>' },
      { type: 'TEXT', value: '/tmp/foo.txt' },
      { type: 'STRING_DELIM', value: '<|"|>' },
      { type: 'TEXT', value: '}' },
      { type: 'TOOL_CALL_END', value: '<tool_call|>' },
    ])
  })

  it('tokenizes a JSON-format tool call (no STRING_DELIM)', () => {
    const input = '<|tool_call>call:read_file{"path":"package.json"}<tool_call|>'
    const tokens = tokenize(input)
    expect(tokens).toEqual([
      { type: 'TOOL_CALL_START', value: '<|tool_call>' },
      { type: 'TEXT', value: 'call:read_file{"path":"package.json"}' },
      { type: 'TOOL_CALL_END', value: '<tool_call|>' },
    ])
  })

  it('tokenizes a thinking block', () => {
    const input = '<|channel>thought\nI need to think<channel|>'
    const tokens = tokenize(input)
    expect(tokens).toEqual([
      { type: 'CHANNEL_START', value: '<|channel>' },
      { type: 'TEXT', value: 'thought\nI need to think' },
      { type: 'CHANNEL_END', value: '<channel|>' },
    ])
  })

  it('tokenizes a full model turn with thinking and tool call', () => {
    const input =
      '<|turn>model\n' +
      '<|channel>thought\nLet me read the file<channel|>' +
      '<|tool_call>call:read_file{path:<|"|>a.txt<|"|>}<tool_call|>' +
      '<turn|>'
    const types = tokenize(input).map(t => t.type)
    expect(types).toEqual([
      'TURN_START', 'TEXT',        // <|turn> + "model\n"
      'CHANNEL_START', 'TEXT', 'CHANNEL_END',  // thinking block
      'TOOL_CALL_START', 'TEXT', 'STRING_DELIM', 'TEXT', 'STRING_DELIM', 'TEXT', 'TOOL_CALL_END',  // tool call
      'TURN_END',
    ])
  })

  it('tokenizes tool response block', () => {
    const input = '<|tool_response>response:read_file{content:<|"|>hello<|"|>}<tool_response|>'
    const tokens = tokenize(input)
    expect(tokens[0].type).toBe('TOOL_RESPONSE_START')
    expect(tokens[tokens.length - 1].type).toBe('TOOL_RESPONSE_END')
  })

  it('handles multiple tool calls in sequence', () => {
    const input =
      '<|tool_call>call:foo{}<tool_call|>' +
      '<|tool_call>call:bar{}<tool_call|>'
    const starts = tokenize(input).filter(t => t.type === 'TOOL_CALL_START')
    expect(starts).toHaveLength(2)
  })

  it('handles tool declaration blocks', () => {
    const input = '<|tool>declaration:read_file{"description":"Read a file"}<tool|>'
    const tokens = tokenize(input)
    expect(tokens[0].type).toBe('TOOL_DECL_START')
    expect(tokens[1]).toEqual({
      type: 'TEXT',
      value: 'declaration:read_file{"description":"Read a file"}',
    })
    expect(tokens[2].type).toBe('TOOL_DECL_END')
  })
})
