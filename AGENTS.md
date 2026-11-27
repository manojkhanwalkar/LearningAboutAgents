# AGENTS.md

Guidance for AI coding agents (Claude Code, Codex, Copilot, Cursor, etc.) working in this repository.

## Project overview

A learning sandbox for building LLM agents in Java with:

- **[LangChain4j](https://github.com/langchain4j/langchain4j)**: model abstraction, AI Services, tools, and chat memory.
- **[LangGraph4j](https://github.com/langgraph4j/langgraph4j)**: stateful, graph-based agent workflows (the Java port of LangGraph).
- **[Ollama](https://ollama.com)**: the only model provider. All models run locally, and no cloud API keys are used.

## Tech stack

| Item | Version |
|---|---|
| Java | 26 (`maven.compiler.release` in `pom.xml`) |
| Build | Maven, via the wrapper (`./mvnw`) |
| LangChain4j | managed by `langchain4j-bom` |
| LangGraph4j | managed by `langgraph4j-bom` |
| Tests | JUnit Jupiter |

Library versions are defined once as properties in `pom.xml`. Add LangChain4j and LangGraph4j modules **without** a `<version>`, because the BOMs manage them.

## Commands

```bash
./mvnw compile                         # compile
./mvnw test                            # unit tests (no Ollama needed)
./mvnw verify                          # tests + JaCoCo coverage report (target/site/jacoco)
./mvnw exec:java                       # run the default example (HelloOllama)
./mvnw exec:java -Dexec.mainClass=com.learning.agents.tools.ToolCallingAgent
./mvnw exec:java -Dexec.mainClass=com.learning.agents.graph.WriterReviewerGraph
./mvnw exec:java -Dexec.args="your prompt here"
```

## Ollama configuration

Read from environment variables in `com.learning.agents.OllamaConfig`:

| Variable | Default |
|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` |
| `OLLAMA_MODEL` | `llama3.1` |
| `LLM_BACKEND` | `placeholder`, read by `com.learning.agents.llm.LlmProvider`; `ollama` switches the agent-pattern samples to a real model |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text`, used by `OllamaConfig.embeddingModel()` |

`tools.ToolCallingAgent` needs a model that supports native **tool calling** (e.g. `llama3.1`, `qwen3`, `mistral`).
The other samples, including all graphs, use plain-text prompts and work with any instruction-following model (e.g. `gemma3`).
Check that the server is up with `curl localhost:11434/api/tags`.

## Layout

```
src/main/java/com/learning/agents/
  OllamaConfig.java          # the single place that builds Ollama models
  llm/                       # LlmProvider (placeholder vs Ollama), ScriptedLlm (canned replies), HashingEmbeddingModel
  chat/                      # prompt -> reply patterns: memory, structured output, guardrails, voting
  tools/                     # LangChain4j AI Services + @Tool examples
  graph/                     # LangGraph4j StateGraph workflows
  rag/                       # retrieval-augmented generation (keyword, embedding and corrective RAG)
  memory/                    # long-term memory that persists across sessions
src/test/java/...            # tests mirror main packages
.claude/skills/              # Claude Code skills (e.g. new-agent-example scaffolds a new sample)
```

## Conventions

- Create models through `OllamaConfig`. Do not hard-code URLs or model names in examples.
- Samples that embed text take a LangChain4j `EmbeddingModel` and get it through
  `LlmProvider.resolveEmbeddings(new HashingEmbeddingModel())`. Embedding-store scores are `(cosine + 1) / 2`, so 0.5 means unrelated.
- New samples take their LLM as a `Function<String, String>`, provide a `static ScriptedLlm placeholder()` with canned
  replies, and get it in `main` through `LlmProvider.resolve(placeholder())`, so they run without Ollama by default.
  A placeholder that must vary between calls (e.g. `SelfConsistencyAgent`'s noisy voter) may return a plain thread-safe `Function`.
- Values kept in graph state must be `Serializable` (records of strings and numbers are fine): LangGraph4j copies
  state between nodes with Java serialization and fails at run time on anything else.
- Accumulating graph-state keys whose entries can legitimately repeat (worker results, agent scratchpads) use
  `Channels.appenderWithDuplicate`; plain `Channels.appender` silently drops a value already in the list.
- Each example is a self-contained class with a `main` method that accepts an optional prompt via `args`.
- Graph nodes take their LLM as a `Function<String, String>` (or similar) so they can be unit-tested with a fake. **Unit tests must not require a running Ollama server.**
- Put graph state in a dedicated `AgentState` subclass with constant keys and typed accessors.
- Use Java records, text blocks, `var`, and other modern language features where they improve clarity.
- Logging goes through SLF4J (`slf4j-simple`, configured in `src/main/resources/simplelogger.properties`).

## Before finishing a change

1. `./mvnw test` passes.
2. New examples are listed in `README.md`.
3. Never commit secrets, `target/`, or IDE workspace files (see `.gitignore`).
