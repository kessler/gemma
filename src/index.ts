export { Gemma } from './gemma.js'

export { Agent, parseToolCalls, hasToolCalls, extractThinking, extractFinalResponse, tokenize, buildPrompt, appendToolCallAndResponse, image, audio, ToolResultImage, ToolResultAudio } from '@kessler/gemma-agent'
export type { Token, TokenType, ModelBackend, GenerateOptions, MediaAttachment, ToolResultValue, Logger, ToolParameterDef, ToolDefinition, ToolCall, ToolResponse, AgentOptions, AgentRunResult, ConversationMessage } from '@kessler/gemma-agent'

export type {
  GemmaOptions,
  DeviceType,
  CompleteOptions,
  CompletionInput,
  ChatMessage,
  ContentItem,
  ProgressInfo,
} from './types.js'

export { MODELS, DEFAULT_MODEL_ID, resolveModelId } from './models.js'
export type { ModelConfig } from './models.js'
