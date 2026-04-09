export type TokenType =
  | 'TOOL_CALL_START'
  | 'TOOL_CALL_END'
  | 'TOOL_RESPONSE_START'
  | 'TOOL_RESPONSE_END'
  | 'STRING_DELIM'
  | 'CHANNEL_START'
  | 'CHANNEL_END'
  | 'TURN_START'
  | 'TURN_END'
  | 'TOOL_DECL_START'
  | 'TOOL_DECL_END'
  | 'EOS'
  | 'END_OF_TURN'
  | 'BOS'
  | 'IMAGE'
  | 'THINK'
  | 'TEXT'

export interface Token {
  type: TokenType
  value: string
}

/**
 * Gemma 4 special token table, sorted by length descending
 * so longer tokens match first (e.g. <|tool_call> before <|tool>).
 */
const SPECIAL_TOKENS: readonly [string, TokenType][] = [
  ['<|tool_response>', 'TOOL_RESPONSE_START'],
  ['<tool_response|>', 'TOOL_RESPONSE_END'],
  ['<|tool_call>', 'TOOL_CALL_START'],
  ['<tool_call|>', 'TOOL_CALL_END'],
  ['<end_of_turn>', 'END_OF_TURN'],
  ['<|channel>', 'CHANNEL_START'],
  ['<channel|>', 'CHANNEL_END'],
  ['<|image|>', 'IMAGE'],
  ['<|think|>', 'THINK'],
  ['<|turn>', 'TURN_START'],
  ['<turn|>', 'TURN_END'],
  ['<|tool>', 'TOOL_DECL_START'],
  ['<tool|>', 'TOOL_DECL_END'],
  ['<|"|>', 'STRING_DELIM'],
  ['<eos>', 'EOS'],
  ['<bos>', 'BOS'],
]

/**
 * Single-pass lexer for Gemma 4 model output.
 * Scans the input left-to-right, emitting special tokens and TEXT runs.
 */
export function tokenize(input: string): Token[] {
  const tokens: Token[] = []
  let pos = 0
  let textStart = 0

  while (pos < input.length) {
    if (input[pos] === '<') {
      let matched = false
      for (const [str, type] of SPECIAL_TOKENS) {
        if (input.startsWith(str, pos)) {
          // Flush accumulated text before this token
          if (pos > textStart) {
            tokens.push({ type: 'TEXT', value: input.slice(textStart, pos) })
          }
          tokens.push({ type, value: str })
          pos += str.length
          textStart = pos
          matched = true
          break
        }
      }
      if (!matched) {
        pos++
      }
    } else {
      pos++
    }
  }

  // Flush trailing text
  if (pos > textStart) {
    tokens.push({ type: 'TEXT', value: input.slice(textStart, pos) })
  }

  return tokens
}
