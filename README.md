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

The unit tests use a fake LLM, so they don't need Ollama.

## Examples

| Example | What it shows | Run |
|---|---|---|
| `chat.HelloOllama` | Single prompt → reply with `OllamaChatModel` | `./mvnw exec:java` |
| `tools.ToolCallingAgent` | LangChain4j AI Service that calls `@Tool` methods, with chat memory | `./mvnw exec:java -Dexec.mainClass=com.learning.agents.tools.ToolCallingAgent` |
| `graph.WriterReviewerGraph` | LangGraph4j `StateGraph` with a writer → reviewer loop and a conditional edge | `./mvnw exec:java -Dexec.mainClass=com.learning.agents.graph.WriterReviewerGraph` |
| `graph.ReActAgentGraph` | ReAct pattern: an explicit Thought → Action → Observation loop with text-parsed tool calls, a step limit, and a final answer (works with any instruction-following model, so native tool calling isn't needed) | `./mvnw exec:java -Dexec.mainClass=com.learning.agents.graph.ReActAgentGraph` |

Pass your own prompt with `-Dexec.args="..."`.

## Configuration

| Env var | Default |
|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` |
| `OLLAMA_MODEL` | `llama3.1` |

```bash
OLLAMA_MODEL=qwen3 ./mvnw exec:java -Dexec.mainClass=com.learning.agents.tools.ToolCallingAgent
```

## Opening in IntelliJ IDEA

Use **File → Open** and select this folder. IntelliJ detects `pom.xml` and imports the project as a Maven project.
