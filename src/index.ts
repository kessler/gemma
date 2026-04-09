export { Gemma } from './gemma.js'
export { Agent } from './agent/agent-loop.js'
export { parseToolCalls, hasToolCalls, extractThinking, extractFinalResponse } from './agent/parser.js'
export { tokenize } from './agent/lexer.js'
export type { Token, TokenType } from './agent/lexer.js'

export type {
  GemmaOptions,
  DeviceType,
  CompleteOptions,
  CompletionInput,
  ChatMessage,
  ContentItem,
  ProgressInfo,
  ToolParameterDef,
  ToolDefinition,
  ToolCall,
  ToolResponse,
  AgentOptions,
  AgentRunResult,
  ConversationMessage,
} from './types.js'

export { MODELS, DEFAULT_MODEL_ID, resolveModelId } from './models.js'
export type { ModelConfig } from './models.js'
