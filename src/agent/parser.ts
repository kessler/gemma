import type { ToolCall } from '../types.js'
import { tokenize, type Token } from './lexer.js'

/**
 * Parse all tool calls from model output.
 * Handles both JSON-format args and Gemma custom <|"|>-delimited args.
 */
export function parseToolCalls(text: string): ToolCall[] {
  const tokens = tokenize(text)
  const calls: ToolCall[] = []
  let i = 0

  while (i < tokens.length) {
    if (tokens[i].type === 'TOOL_CALL_START') {
      i++
      const inner: Token[] = []
      while (i < tokens.length && tokens[i].type !== 'TOOL_CALL_END') {
        inner.push(tokens[i])
        i++
      }
      if (i < tokens.length) i++ // skip TOOL_CALL_END

      const call = parseToolCallBody(inner)
      if (call) calls.push(call)
    } else {
      i++
    }
  }

  return calls
}

export function hasToolCalls(text: string): boolean {
  return text.includes('<|tool_call>')
}

/**
 * Extract thinking content and the remaining text (with special tokens intact).
 * The `rest` preserves all special tokens so it can be fed into parseToolCalls.
 */
export function extractThinking(text: string): { thinking: string; rest: string } {
  const tokens = tokenize(text)
  let thinking = ''
  let inChannel = false
  let channelContent = ''
  const restParts: string[] = []

  for (const token of tokens) {
    if (token.type === 'CHANNEL_START') {
      inChannel = true
      channelContent = ''
      continue
    }
    if (token.type === 'CHANNEL_END') {
      inChannel = false
      if (channelContent.startsWith('thought\n')) {
        thinking = channelContent.slice('thought\n'.length).trim()
      } else if (channelContent.startsWith('thought')) {
        thinking = channelContent.slice('thought'.length).trim()
      } else {
        thinking = channelContent.trim()
      }
      continue
    }
    if (inChannel) {
      channelContent += token.value
      continue
    }
    restParts.push(token.value)
  }

  return { thinking, rest: restParts.join('') }
}

/**
 * Extract the final user-facing response by stripping all special tokens,
 * tool call/response blocks, thinking blocks, and turn markers.
 */
export function extractFinalResponse(text: string): string {
  const tokens = tokenize(text)
  const parts: string[] = []
  let depth = 0
  let afterTurnStart = false

  for (const token of tokens) {
    // Track block-level nesting (tool calls, responses, channels, declarations)
    if (
      token.type === 'TOOL_CALL_START' ||
      token.type === 'TOOL_RESPONSE_START' ||
      token.type === 'CHANNEL_START' ||
      token.type === 'TOOL_DECL_START'
    ) {
      depth++
      continue
    }
    if (
      token.type === 'TOOL_CALL_END' ||
      token.type === 'TOOL_RESPONSE_END' ||
      token.type === 'CHANNEL_END' ||
      token.type === 'TOOL_DECL_END'
    ) {
      depth = Math.max(0, depth - 1)
      continue
    }
    if (depth > 0) continue

    if (token.type === 'TURN_START') {
      afterTurnStart = true
      continue
    }

    // Drop all remaining special tokens
    if (
      token.type === 'TURN_END' ||
      token.type === 'EOS' ||
      token.type === 'END_OF_TURN' ||
      token.type === 'BOS' ||
      token.type === 'IMAGE' ||
      token.type === 'THINK' ||
      token.type === 'STRING_DELIM'
    ) {
      continue
    }

    if (token.type === 'TEXT') {
      let value = token.value
      if (afterTurnStart) {
        afterTurnStart = false
        // Strip the role label that immediately follows <|turn>
        for (const role of ['model', 'system', 'user']) {
          if (value.startsWith(role)) {
            value = value.slice(role.length).trimStart()
            break
          }
        }
      }
      if (value) parts.push(value)
    }
  }

  return parts.join('').trim()
}

// ---- Private helpers ----

const CALL_PREFIX = 'call:'
const NULL_MARKER = '\x00'

function parseToolCallBody(tokens: Token[]): ToolCall | null {
  // Join only TEXT values to get the textual content
  const fullText = tokens
    .filter(t => t.type === 'TEXT')
    .map(t => t.value)
    .join('')

  if (!fullText.startsWith(CALL_PREFIX)) return null

  const braceIndex = fullText.indexOf('{')
  if (braceIndex === -1) return null

  const name = fullText.slice(CALL_PREFIX.length, braceIndex)

  // Try JSON first — works for standard {"key":"value"} format
  const argsText = fullText.slice(braceIndex)
  try {
    return { name, arguments: JSON.parse(argsText) }
  } catch {
    // Fall back to Gemma custom format (unquoted keys, STRING_DELIM strings)
  }

  return { name, arguments: parseGemmaArgs(tokens) }
}

/**
 * Parse Gemma custom arg format: {key:<|"|>value<|"|>,key2:123}
 *
 * Strategy: rebuild the text with NULL_MARKER replacing STRING_DELIM,
 * then walk the key:value pairs. Marker-wrapped values are strings,
 * bare values are cast to their natural type.
 */
function parseGemmaArgs(tokens: Token[]): Record<string, unknown> {
  let rebuilt = ''
  for (const token of tokens) {
    if (token.type === 'STRING_DELIM') {
      rebuilt += NULL_MARKER
    } else if (token.type === 'TEXT') {
      rebuilt += token.value
    }
  }

  const start = rebuilt.indexOf('{')
  const end = rebuilt.lastIndexOf('}')
  if (start === -1 || end === -1 || start >= end) return {}

  const body = rebuilt.slice(start + 1, end)
  const args: Record<string, unknown> = {}
  let pos = 0

  while (pos < body.length) {
    // Skip commas and whitespace
    while (pos < body.length && (body[pos] === ',' || body[pos] === ' ')) pos++
    if (pos >= body.length) break

    // Read key (everything up to ':')
    const keyStart = pos
    while (pos < body.length && body[pos] !== ':') pos++
    const key = body.slice(keyStart, pos).trim()
    if (!key || pos >= body.length) break
    pos++ // skip ':'

    if (pos < body.length && body[pos] === NULL_MARKER) {
      // String value wrapped in STRING_DELIM markers
      pos++ // skip opening marker
      const valStart = pos
      while (pos < body.length && body[pos] !== NULL_MARKER) pos++
      args[key] = body.slice(valStart, pos)
      if (pos < body.length) pos++ // skip closing marker
    } else {
      // Bare value — read until comma or end
      const valStart = pos
      while (pos < body.length && body[pos] !== ',') pos++
      args[key] = castValue(body.slice(valStart, pos).trim())
    }
  }

  return args
}

function castValue(value: string): unknown {
  if (value === 'true') return true
  if (value === 'false') return false
  if (value === 'null') return null

  const num = Number(value)
  if (!isNaN(num) && value !== '') return num

  return value
}
