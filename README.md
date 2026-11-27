# LearningAboutAgents

Hands-on experiments with LLM agents in Java, using **LangChain4j** and **LangGraph4j** with models served locally by **Ollama**.

## Prerequisites

- JDK 26
- [Ollama](https://ollama.com) running locally, with a model that supports tool calling:
  ```bash
  ollama pull llama3.1
  ```
- Maven is not required, because the Maven wrapper (`./mvnw`) is included.

## Build and test

```bash
./mvnw test
```

The unit tests use a fake LLM, so they don't need Ollama. `./mvnw verify` also writes a JaCoCo coverage report to
`target/site/jacoco/index.html`.

## Examples

New to agents? Read [**Agent concepts, one sample at a time**](docs/agent-concepts.md). It explains the idea behind
each sample, how the code implements it, what to look for when you run it, and what to try next. Its
[suggested reading order](docs/agent-concepts.md#suggested-reading-order) groups the chapters into eight stages, from
"what an agent is" to running agents in production, with a question to check your understanding after each stage.

| Example | What it shows | Run |
|---|---|---|
| `chat.HelloOllama` | Single prompt → reply with `OllamaChatModel` | `./mvnw exec:java` |
| `tools.ToolCallingAgent` | LangChain4j AI Service that calls `@Tool` methods, with chat memory | `./mvnw exec:java -Dexec.mainClass=com.learning.agents.tools.ToolCallingAgent` |
| `graph.WriterReviewerGraph` | LangGraph4j `StateGraph` with a writer → reviewer loop and a conditional edge | `./mvnw exec:java -Dexec.mainClass=com.learning.agents.graph.WriterReviewerGraph` |
| `graph.ReActAgentGraph` | ReAct pattern: an explicit Thought → Action → Observation loop with text-parsed tool calls, a step limit, and a final answer (works with any instruction-following model, so native tool calling isn't needed) | `./mvnw exec:java -Dexec.mainClass=com.learning.agents.graph.ReActAgentGraph` |

Pass your own prompt with `-Dexec.args="..."`.

### Agent-pattern samples (placeholder LLM)

These ten samples run **without Ollama**: each one ships a scripted placeholder LLM, so you can step through the pattern
and read its output right away. Set `LLM_BACKEND=ollama` to send the same prompts to a real model. They use plain-text
prompts only, with no native tool calling, so any instruction-following model works, including Gemma.

Suggested reading order, simplest first:

| # | Example | Concept | Run |
|---|---|---|---|
| 1 | `graph.PromptChainingGraph` | **Prompt chaining**: a fixed outline → draft → polish chain, with a plain-code gate that retries a bad outline | `./mvnw exec:java -Dexec.mainClass=com.learning.agents.graph.PromptChainingGraph` |
| 2 | `graph.RouterGraph` | **Routing**: a classifier call picks a specialist prompt; unexpected labels fall back to a safe default | `./mvnw exec:java -Dexec.mainClass=com.learning.agents.graph.RouterGraph` |
| 3 | `chat.StructuredOutputAgent` | **Structured output**: ask for JSON, parse and validate it into a record, and send the error back so the model can repair it | `./mvnw exec:java -Dexec.mainClass=com.learning.agents.chat.StructuredOutputAgent` |
| 4 | `chat.ConversationMemoryAgent` | **Memory**: keep recent turns verbatim and fold older ones into an LLM-written running summary | `./mvnw exec:java -Dexec.mainClass=com.learning.agents.chat.ConversationMemoryAgent` |
| 5 | `rag.RagAgent` | **RAG**: TF-IDF retrieval over a small handbook, grounded answers with `[doc-id]` citations, and "I don't know" with no LLM call when nothing matches | `./mvnw exec:java -Dexec.mainClass=com.learning.agents.rag.RagAgent` |
| 6 | `chat.GuardrailsAgent` | **Guardrails**: input checks (length, prompt injection, blocked topics) before the call, output checks (PII redaction, system-prompt leaks) after it | `./mvnw exec:java -Dexec.mainClass=com.learning.agents.chat.GuardrailsAgent` |
| 7 | `chat.SelfConsistencyAgent` | **Self-consistency and parallel calls**: sample the same question 5 times on virtual threads and take a majority vote (raise the temperature for real models) | `./mvnw exec:java -Dexec.mainClass=com.learning.agents.chat.SelfConsistencyAgent` |
| 8 | `graph.OrchestratorWorkersGraph` | **Orchestrator-workers**: the LLM writes the plan at run time, one worker per subtask, then a synthesizer merges the results | `./mvnw exec:java -Dexec.mainClass=com.learning.agents.graph.OrchestratorWorkersGraph` |
| 9 | `graph.SupervisorGraph` | **Multi-agent supervisor**: researcher, writer and critic agents share a scratchpad, and a supervisor picks who acts next until `FINISH` | `./mvnw exec:java -Dexec.mainClass=com.learning.agents.graph.SupervisorGraph` |
| 10 | `graph.HumanApprovalGraph` | **Human in the loop**: a checkpointer plus `interruptBefore` pauses the graph so you approve the action on the console before it runs | `./mvnw exec:java -Dexec.mainClass=com.learning.agents.graph.HumanApprovalGraph` |

### More agent patterns

Same rules as above: placeholder by default, `LLM_BACKEND=ollama` for a real model. Samples that embed text use
`HashingEmbeddingModel` as a placeholder (it matches shared words only) and `OLLAMA_EMBEDDING_MODEL` with Ollama.

**Retrieval and memory**

| Example | Concept | Run |
|---|---|---|
| `rag.EmbeddingRagAgent` | **Embedding RAG**: LangChain4j document splitter, embedding model and `InMemoryEmbeddingStore`, plus LLM query rewriting and reranking | `./mvnw exec:java -Dexec.mainClass=com.learning.agents.rag.EmbeddingRagAgent` |
| `rag.CorrectiveRagGraph` | **Corrective RAG**: an LLM grades retrieved chunks, and if none help, rewrites the query and searches again before answering or giving up | `./mvnw exec:java -Dexec.mainClass=com.learning.agents.rag.CorrectiveRagGraph` |
| `memory.LongTermMemoryAgent` | **Long-term memory**: facts extracted from user messages are saved to a JSON file and recalled by similarity in a later session, with secret filtering and `forget` | `./mvnw exec:java -Dexec.mainClass=com.learning.agents.memory.LongTermMemoryAgent` |

**Reasoning and planning**

| Example | Concept | Run |
|---|---|---|
| `reasoning.PlanExecuteGraph` | **Plan-and-execute with replanning**: plan, do one step, then rewrite the remaining plan from what was learned (or finish with `FINAL:`) | `./mvnw exec:java -Dexec.mainClass=com.learning.agents.reasoning.PlanExecuteGraph` |
| `reasoning.ReflexionGraph` | **Reflexion**: a code checker grades each attempt, the model writes a lesson after each failure, and all lessons feed the next attempt | `./mvnw exec:java -Dexec.mainClass=com.learning.agents.reasoning.ReflexionGraph` |
| `reasoning.TreeOfThoughtsAgent` | **Tree of Thoughts**: beam search on the Game of 24; the model proposes and rates steps, code verifies the arithmetic | `./mvnw exec:java -Dexec.mainClass=com.learning.agents.reasoning.TreeOfThoughtsAgent` |

**LangGraph4j features**

| Example | Concept | Run |
|---|---|---|
| `langgraph.ParallelBranchesGraph` | **Parallel branches**: fan-out to three analysts and fan-in to a judge; shows that branches only overlap with `addParallelNodeExecutor` | `./mvnw exec:java -Dexec.mainClass=com.learning.agents.langgraph.ParallelBranchesGraph` |
| `langgraph.SubgraphGraph` | **Subgraphs**: a compiled research graph (with its own loop) used as one node of a report graph | `./mvnw exec:java -Dexec.mainClass=com.learning.agents.langgraph.SubgraphGraph` |
| `langgraph.TimeTravelGraph` | **Time travel**: checkpoint history, replay from a past step, and fork with changed state | `./mvnw exec:java -Dexec.mainClass=com.learning.agents.langgraph.TimeTravelGraph` |
| `langgraph.StreamingAgent` | **Streaming**: node events with `graph.stream(...)` and token-by-token output (`OllamaStreamingChatModel` with Ollama) | `./mvnw exec:java -Dexec.mainClass=com.learning.agents.langgraph.StreamingAgent` |

**Tools and multi-agent**

| Example | Concept | Run |
|---|---|---|
| `tools.mcp.McpAgent` | **MCP tools**: lists and calls tools on an MCP server (a tiny in-process one over real Streamable HTTP), with an allow-list and fenced, size-capped results | `./mvnw exec:java -Dexec.mainClass=com.learning.agents.tools.mcp.McpAgent` |
| `tools.ToolRetrievalAgent` | **Tool retrieval**: embeds a 12-tool catalog and shows the model only the 3 most relevant tools per question | `./mvnw exec:java -Dexec.mainClass=com.learning.agents.tools.ToolRetrievalAgent` |
| `multiagent.HandoffGraph` | **Handoffs (swarm)**: agents pass the conversation directly to each other, with no supervisor, under a handoff limit | `./mvnw exec:java -Dexec.mainClass=com.learning.agents.multiagent.HandoffGraph` |
| `multiagent.DebateGraph` | **Debate**: pro and con argue for two rounds, then a judge picks a winner (parsed strictly) | `./mvnw exec:java -Dexec.mainClass=com.learning.agents.multiagent.DebateGraph` |

**Production concerns**

| Example | Concept | Run |
|---|---|---|
| `ops.EvalHarness` | **Evaluation**: a golden dataset scored by a code check and a strict LLM judge; flags where they disagree; compares two candidates | `./mvnw exec:java -Dexec.mainClass=com.learning.agents.ops.EvalHarness` |
| `ops.ModelCascade` | **Cascade and resilience**: small model first with self-rated confidence, escalation to a large model, deadlines, full-jitter retries of recoverable errors only, a fallback model | `./mvnw exec:java -Dexec.mainClass=com.learning.agents.ops.ModelCascade` |
| `ops.ObservabilityDemo` | **Observability and budgets**: the `ObservedLlm` decorator traces latency, sizes and tokens (no content) and a token budget stops a runaway agent | `./mvnw exec:java -Dexec.mainClass=com.learning.agents.ops.ObservabilityDemo` |
| `ops.SemanticCache` | **Semantic caching**: reuse answers for similar questions (LRU + TTL); shows how a loose threshold returns a wrong answer | `./mvnw exec:java -Dexec.mainClass=com.learning.agents.ops.SemanticCache` |

Running them against your Ollama server, for example on another machine on your network:

```bash
LLM_BACKEND=ollama OLLAMA_BASE_URL=http://192.168.1.50:11434 OLLAMA_MODEL=gemma3 \
  ./mvnw exec:java -Dexec.mainClass=com.learning.agents.rag.RagAgent
```

The placeholder replies are keyed to the default prompts, so with the placeholder, a custom `-Dexec.args` prompt mostly
gets the fallback reply. Use `LLM_BACKEND=ollama` to try your own prompts.

## Configuration

| Env var | Default |
|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` |
| `OLLAMA_MODEL` | `llama3.1` |
| `LLM_BACKEND` | `placeholder` (the agent-pattern samples only; `ollama` uses the settings above) |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` (embedding samples, with `LLM_BACKEND=ollama`; `ollama pull nomic-embed-text`) |
| `CASCADE_SMALL_MODEL`, `CASCADE_LARGE_MODEL`, `CASCADE_FALLBACK_MODEL` | `OLLAMA_MODEL` (`ModelCascade` tiers, with `LLM_BACKEND=ollama`) |
| `AGENT_MEMORY_FILE` | a temporary file (`LongTermMemoryAgent`; set it to keep memory between runs) |

```bash
OLLAMA_MODEL=qwen3 ./mvnw exec:java -Dexec.mainClass=com.learning.agents.tools.ToolCallingAgent
```

## Opening in IntelliJ IDEA

Use **File → Open** and select this folder. IntelliJ detects `pom.xml` and imports the project as a Maven project.
