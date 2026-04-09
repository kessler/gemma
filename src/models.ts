export interface ModelConfig {
  id: string
  hfModelId: string
  label: string
  downloadSize: string
  contextLimit: number
}

export const MODELS: Record<string, ModelConfig> = {
  'gemma-4-e2b': {
    id: 'gemma-4-e2b',
    hfModelId: 'onnx-community/gemma-4-E2B-it-ONNX',
    label: 'Gemma 4 E2B',
    downloadSize: '~500MB',
    contextLimit: 128_000,
  },
  'gemma-4-e4b': {
    id: 'gemma-4-e4b',
    hfModelId: 'onnx-community/gemma-4-E4B-it-ONNX',
    label: 'Gemma 4 E4B',
    downloadSize: '~1.5GB',
    contextLimit: 128_000,
  },
}

export const DEFAULT_MODEL_ID = 'gemma-4-e4b'

export function resolveModelId(model: string): { hfModelId: string; contextLimit: number } {
  const config = MODELS[model]
  if (config) {
    return { hfModelId: config.hfModelId, contextLimit: config.contextLimit }
  }
  // Treat as a custom HuggingFace model ID
  return { hfModelId: model, contextLimit: 128_000 }
}
