import type { ToolDefinition, ToolCall, ToolResponse, ConversationMessage } from '../types.js'

function formatToolDeclaration(tool: ToolDefinition): string {
  const schema: Record<string, unknown> = {
    description: tool.description,
  }
  if (tool.parameters) {
    schema.parameters = tool.parameters
  }
  return `<|tool>declaration:${tool.name}${JSON.stringify(schema)}<tool|>`
}

function formatToolResponse(response: ToolResponse): string {
  const entries = Object.entries(response.result as Record<string, unknown>)
    .map(([k, v]) => {
      if (typeof v === 'string') return `${k}:<|"|>${v}<|"|>`
      return `${k}:${v}`
    })
    .join(',')
  return `response:${response.name}{${entries}}<tool_response|>`
}

function formatToolCallArgs(call: ToolCall): string {
  return Object.entries(call.arguments)
    .map(([k, v]) => {
      if (typeof v === 'string') return `${k}:<|"|>${v}<|"|>`
      return `${k}:${v}`
    })
    .join(',')
}

export function buildPrompt(
  systemPrompt: string,
  tools: ToolDefinition[],
  history: ConversationMessage[],
  enableThinking: boolean,
): string {
  const parts: string[] = []

  const thinkToken = enableThinking ? '<|think|>' : ''
  const toolDeclarations = tools.map(formatToolDeclaration).join('')
  parts.push(`<|turn>system\n${thinkToken}${systemPrompt}${toolDeclarations}<turn|>`)

  for (const msg of history) {
    if (msg.role === 'user') {
      parts.push(`<|turn>user\n${msg.content}<turn|>`)
    }

    if (msg.role === 'model') {
      if (msg.toolCalls && msg.toolResponses) {
        const callStr = msg.toolCalls
          .map(call => `<|tool_call>call:${call.name}{${formatToolCallArgs(call)}}<tool_call|>`)
          .join('')

        const respStr = msg.toolResponses.map(formatToolResponse).join('')
        parts.push(`<|turn>model\n${callStr}<|tool_response>${respStr}`)

        if (msg.content) {
          parts.push(`${msg.content}<turn|>`)
        }
      } else {
        parts.push(`<|turn>model\n${msg.content}<turn|>`)
      }
    }
  }

  parts.push('<|turn>model')
  return parts.join('\n')
}

export function appendToolCallAndResponse(
  currentPrompt: string,
  calls: ToolCall[],
  responses: ToolResponse[],
): string {
  const callStr = calls
    .map(call => `<|tool_call>call:${call.name}{${formatToolCallArgs(call)}}<tool_call|>`)
    .join('')

  const respStr = responses.map(formatToolResponse).join('')
  return `${currentPrompt}\n${callStr}<|tool_response>${respStr}`
}
