# Changelog

## 1.0.0

### Fixed

- Agent tool call parsing now handles both JSON-format args (`{"path":"file.json"}`) and Gemma custom format (`{path:<|"|>file.json<|"|>}`). Previously, JSON args from ONNX models were silently dropped.

### Changed

- Replaced regex-based tool parser with a proper lexer + parser (`src/agent/lexer.ts`, `src/agent/parser.ts`). No regexes in the tool parsing pipeline.

### Added

- `tokenize()` function exported for low-level access to Gemma 4's special token stream.
- `Token` and `TokenType` types exported.
- Integration tests now cover both `gemma-4-e2b` and `gemma-4-e4b` models across all features (text, chat, streaming, thinking, image, audio, agent).
