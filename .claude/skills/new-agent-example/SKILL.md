---
name: new-agent-example
description: Scaffold a new runnable sample agent in this repo (LangChain4j AI Service with tools, or a LangGraph4j StateGraph workflow) on Ollama, with a fake-LLM unit test and a README entry. Use when the user asks to create, add, or scaffold a new agent example, demo, or sample (e.g. "add a ReAct agent", "make a router graph example", "show RAG with tools").
argument-hint: "<what the agent should demonstrate>"
---

# New agent example

Create a new self-contained sample agent that demonstrates: **$ARGUMENTS**

If `$ARGUMENTS` is empty, ask the user what concept the example should demonstrate before writing code.

## 1. Pick the style

| The concept is mostly about... | Style | Package | Model after |
|---|---|---|---|
| A single prompt → reply, prompt templates, structured output | Chat | `chat/` | `chat/HelloOllama.java` |
| An LLM calling Java methods, chat memory, AI Services | Tools | `tools/` | `tools/ToolCallingAgent.java` + `tools/Calculator.java` |
| Multiple steps, loops, routing, branching, multi-agent handoff | Graph | `graph/` | `graph/WriterReviewerGraph.java` + `graph/WriterState.java` |

Read the reference files for the chosen style before writing anything, and match their structure, Javadoc density, and naming.

## 2. Write the code

All paths are under `src/main/java/com/learning/agents/`.

Rules for every style:
- Get models only from `OllamaConfig.chatModel()`. Never hard-code a base URL or model name.
- Use one class with `public static void main(String[] args)`. If `args` is non-empty, join it into the prompt; otherwise use a sensible default prompt that shows off the concept.
- Add a class-level Javadoc that explains in 1-3 sentences what the example teaches. For graphs, include an ASCII diagram of the nodes and edges.
- Use modern Java (records, text blocks, `var`, switch expressions) where it makes the code clearer.
- If you need a new dependency, add it to `pom.xml` **without** a `<version>` when a LangChain4j or LangGraph4j BOM manages it.

**Tools style**
- Put the tools in their own class, with `@Tool("clear description")` and `@P("...")` on each parameter, because the model reads those descriptions.
- Define the assistant as a nested `interface` with an `@SystemMessage`, and build it with `AiServices.builder(...)`.
- Make the tool logic a plain method that can be unit-tested directly.

**Graph style**
- Add a dedicated `XxxState extends AgentState` with `public static final String` keys, a `SCHEMA` map (use `Channels.appender` for accumulating keys such as logs), and typed accessors that return `Optional` or defaults.
- Expose `public static CompiledGraph<XxxState> build(Function<String, String> llm)` so tests can inject a fake LLM. In `main`, pass `model::chat`.
- Use `node_async` / `edge_async`. Add a guard such as `MAX_...` to every loop so it always terminates, even with a misbehaving model.
- Keep routing decisions in small package-private static methods so they can be tested on their own.

## 3. Write the test

Mirror the package under `src/test/java/com/learning/agents/`, as `XxxTest.java`.

- **Tests must never need Ollama.** Use a lambda as the fake LLM that returns canned replies based on the prompt (see `WriterReviewerGraphTest`).
- Cover the happy path and at least one edge case: a loop limit, an alternate route, or a tool's edge input.
- For tool examples, unit-test the tool class directly. Do not test the AI Service wiring, because that needs a real model.

## 4. Document it

Add a row to the **Examples** table in `README.md`:

```
| `pkg.ClassName` | What it shows | `./mvnw exec:java -Dexec.mainClass=com.learning.agents.pkg.ClassName` |
```

If the example needs a specific kind of model (for example tool calling or JSON mode), say so in that row.

## 5. Verify

1. Run `./mvnw -q test`. It must pass.
2. If Ollama is reachable (`curl -s localhost:11434/api/tags`), run the example once with `./mvnw -q exec:java -Dexec.mainClass=...` and show the user the output. If Ollama isn't reachable, say that the live run was skipped.
3. Report the new files, how to run the example, and what to look for in its output.
