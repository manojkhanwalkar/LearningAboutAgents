# Agent concepts, one sample at a time

This guide walks through every sample in the repository. Each chapter explains one idea behind LLM agents, shows
how the sample implements it, tells you what to look for when you run it, and suggests changes to try. It ends with
what tends to go wrong in real systems. A *Going deeper* section in each chapter explains why the technique
works, where it breaks and what it costs. Read it next to the code: every chapter names the class to open.

- [How the samples are built](#how-the-samples-are-built)
- [Ideas that run through every chapter](#ideas-that-run-through-every-chapter)
- [Suggested reading order](#suggested-reading-order)
- Foundations
  1. [Calling a model](#1-calling-a-model--chathelloollama)
  2. [Native tool calling](#2-native-tool-calling--toolstoolcallingagent)
  3. [Your first graph: evaluator and optimizer](#3-your-first-graph-evaluator-and-optimizer--graphwriterreviewergraph)
  4. [ReAct: reason, act, observe](#4-react-reason-act-observe--graphreactagentgraph)
- Workflow patterns
  5. [Prompt chaining](#5-prompt-chaining--graphpromptchaininggraph)
  6. [Routing](#6-routing--graphroutergraph)
  7. [Orchestrator and workers](#7-orchestrator-and-workers--graphorchestratorworkersgraph)
  8. [Multi-agent supervisor](#8-multi-agent-supervisor--graphsupervisorgraph)
  9. [Human in the loop](#9-human-in-the-loop--graphhumanapprovalgraph)
- Making single calls reliable
  10. [Structured output](#10-structured-output--chatstructuredoutputagent)
  11. [Conversation memory](#11-conversation-memory--chatconversationmemoryagent)
  12. [Retrieval-augmented generation](#12-retrieval-augmented-generation--ragragagent)
  13. [Guardrails](#13-guardrails--chatguardrailsagent)
  14. [Self-consistency and parallel calls](#14-self-consistency-and-parallel-calls--chatselfconsistencyagent)
- Retrieval and memory
  15. [Embedding RAG with rewriting and reranking](#15-embedding-rag-with-rewriting-and-reranking--ragembeddingragagent)
  16. [Corrective RAG](#16-corrective-rag--ragcorrectiveraggraph)
  17. [Long-term memory](#17-long-term-memory--memorylongtermmemoryagent)
- Reasoning and planning
  18. [Plan-and-execute with replanning](#18-plan-and-execute-with-replanning--reasoningplanexecutegraph)
  19. [Reflexion](#19-reflexion--reasoningreflexiongraph)
  20. [Tree of Thoughts](#20-tree-of-thoughts--reasoningtreeofthoughtsagent)
- LangGraph4j features
  21. [Parallel branches](#21-parallel-branches--langgraphparallelbranchesgraph)
  22. [Subgraphs](#22-subgraphs--langgraphsubgraphgraph)
  23. [Time travel with checkpoints](#23-time-travel-with-checkpoints--langgraphtimetravelgraph)
  24. [Streaming](#24-streaming--langgraphstreamingagent)
- Tools and multi-agent
  25. [Tools from an MCP server](#25-tools-from-an-mcp-server--toolsmcpmcpagent)
  26. [Tool retrieval](#26-tool-retrieval--toolstoolretrievalagent)
  27. [Handoffs between agents](#27-handoffs-between-agents--multiagenthandoffgraph)
  28. [Debate](#28-debate--multiagentdebategraph)
- Production concerns
  29. [Evaluating agents](#29-evaluating-agents--opsevalharness)
  30. [Model cascade and resilience](#30-model-cascade-and-resilience--opsmodelcascade)
  31. [Observability and token budgets](#31-observability-and-token-budgets--opsobservedllm)
  32. [Semantic caching](#32-semantic-caching--opssemanticcache)
- [Glossary](#glossary)

---

## How the samples are built

**What an "agent" means here.** An LLM is a function from text to text. It has no memory, cannot act, and cannot
check its own work. An *agent* is ordinary code wrapped around that function. The code decides what to put in the
prompt, what to do with the reply, and whether to call the model again. Every sample is a different answer to those
three questions.

**The LLM is just a `Function<String, String>`.** Apart from chapters 1–4, every sample takes its model as a plain
function, prompt in and reply out, and its `main` gets that function through `LlmProvider`. This has two benefits:

- *Tests and placeholders.* A lambda can stand in for the model, so every sample has unit tests and runs without a
  server. Each sample ships a `placeholder()`: usually a `ScriptedLlm` with canned replies keyed to its own prompts.
  `SelfConsistencyAgent` uses a plain function instead, because its replies must vary between calls.
- *Swapping models is configuration.* `LlmProvider.resolve(placeholder())` returns the placeholder unless
  `LLM_BACKEND=ollama` is set. In that case it returns a real Ollama model, picked by `OLLAMA_BASE_URL` and
  `OLLAMA_MODEL`. None of these samples need native tool calling, so small open models such as Gemma work.

```bash
./mvnw exec:java -Dexec.mainClass=com.learning.agents.graph.RouterGraph            # placeholder
LLM_BACKEND=ollama OLLAMA_BASE_URL=http://<box>:11434 OLLAMA_MODEL=gemma3 \
  ./mvnw exec:java -Dexec.mainClass=com.learning.agents.graph.RouterGraph          # real model
```

The placeholder replies only match the default prompts, so try your own prompts (`-Dexec.args="..."`) with a real
model.

**Chapters 1–4 are the exception.** These original samples call Ollama directly from `main` and have no placeholder,
so running them needs a server: `HelloOllama`, `ToolCallingAgent`, `WriterReviewerGraph` and `ReActAgentGraph`.
`ToolCallingAgent` also needs a model with native tool calling, such as `llama3.1` or `qwen3`. The two graphs still
take a `Function` in `build(...)`, so their tests run on fakes.

**Embeddings.** Samples that search by meaning take a LangChain4j `EmbeddingModel`. With the placeholder backend
that is `HashingEmbeddingModel`, which hashes words into a 512-number vector, so texts that share words come out
similar. With `LLM_BACKEND=ollama` it is `OLLAMA_EMBEDDING_MODEL` (default `nomic-embed-text`), which also links
words with similar meanings.

**Graphs in five lines.** The `graph/` samples use [LangGraph4j](https://github.com/langgraph4j/langgraph4j):

- **State**: a typed map shared by all steps. Each sample has its own `XxxState extends AgentState` with key constants
  and accessors. A key is either *overwritten* by each update or *appended to* (a "channel", such as
  `Channels.appender`).
- **Nodes**: steps. Each one reads the state and returns a map of updates.
- **Edges**: what runs next. A plain edge always goes to the same node. A *conditional* edge calls a small Java
  function that picks the next node from the state.
- **Loops** are edges that point backwards. Every loop in this repository has a hard limit (`MAX_...`), so a confused
  model can never make it run forever.
- **Routing decisions are code**, not model output taken on trust. The model's reply is parsed and checked against
  the allowed choices, and anything unexpected takes a safe default.

Two LangGraph4j details that trip people up:
- `Channels.appender` silently drops a value that is already in the list, so keys that may repeat use
  `appenderWithDuplicate`.
- Every value stored in state must be `Serializable`.

## Ideas that run through every chapter

The 32 chapters are variations on a handful of ideas. Once you can see these, each new pattern reads as a
recombination of them rather than something to memorise.

**1. The model predicts text; it doesn't look things up.** An LLM turns the prompt into tokens, computes a
probability for every possible next token, picks one, appends it, and repeats. Three consequences follow:
- Everything the model "knows" in a call comes from one of two places: its *weights* (frozen at training time,
  fuzzy, impossible to cite) or the *prompt* (fresh, specific, checkable). RAG (12, 15, 16), memory (11, 17) and
  tool results (2, 4, 25) are all ways of moving knowledge out of the weights and into the prompt.
- A fluent wrong answer and a right one come out of the same process. Confident tone tells you nothing.
- Each token is conditioned on the tokens before it, so an early mistake is carried forward rather than noticed.
  Self-consistency (14), Reflexion (19) and Tree of Thoughts (20) all exist to escape that.

**2. The context window is the only working memory.** The model is stateless: every call rebuilds its whole world
from the prompt. So most of agent design is *context engineering*, which means deciding what goes into the prompt, in
what order, fenced how, and what gets left out. More is not better. Every token costs money and time, and models
pay less attention to the middle of long prompts than to their start and end ("lost in the middle", Liu et al.,
2023). Chapters 11, 15, 17, 26 and 31 are about keeping the context small and relevant.

**3. Checking is easier than generating.** Confirming that `13 - 9 = 4` is trivial; finding the path to 24 is not.
Confirming that JSON parses is trivial; writing a correct review summary is not. This *generator–verifier gap* is
behind most reliability patterns: let the model generate, and put a checker after it. The stronger the checker, the
more a retry loop is worth. In order of strength:

| Checker | Example chapters | Why it is that strong |
|---|---|---|
| Code that computes the answer | 19 (`checkPrimes`), 20 (`parseStep`) | Exact and repeatable |
| Code that checks the shape | 5 (gate), 10 (schema), 6 (allow-list) | Exact, but blind to meaning |
| A separate model call as judge | 16 (grader), 28 (judge), 29 (LLM judge) | Understands meaning, has biases |
| The same model critiquing itself | 3 (reviewer), 30 (self-rated confidence) | Shares the generator's blind spots |

**4. Errors compound.** If each step of a run is right 95% of the time, a 10-step run is fully right only about 60%
of the time (0.95¹⁰ ≈ 0.60). Long autonomous runs fail for this reason, not because any one step is bad. The
remedies recur throughout the guide: use fewer steps, check between steps so an error is caught at step 2 rather
than step 10 (5, 16), replan when reality diverges (18), and put a person in front of irreversible steps (9).

**5. Choose the least autonomy that does the job.** The patterns sit on a spectrum of who decides what happens next:

```
fixed steps ──── code picks a branch ──── model plans within limits ──── model decides every step
 chaining (5)      routing (6)              orchestrator (7), plan (18)    ReAct (4), supervisor (8), handoffs (27)
```

Moving right buys flexibility and pays for it with predictability, cost, latency and testability. Anthropic's
*Building effective agents* (2024) draws the same line between *workflows*, where code fixes the path, and
*agents*, where the model chooses it. Start at the left and move right only when the task really varies at run time.

**6. The model proposes; code decides.** Treat model output like user input, as untrusted text to be parsed and
checked. In every sample, a route is mapped onto an allow-list with a safe default, a loop has a hard cap, and the
result of an action comes from code, never from the model saying it happened. Untrusted text going *into* the
prompt (from users, documents, tool results, memory or other agents) is fenced and escaped. Fencing lowers the risk
but can't remove it: a model reads instructions and data through the same channel, which is the root cause of
prompt injection. That is why the real defence is limiting what the agent *can* do.

**7. Cost is calls × tokens; latency is calls in sequence.** You can price any pattern on paper before running it:
count the model calls per request, estimate each prompt's size, and note which calls have to wait for each other.
Calls in sequence add up their latency. Parallel calls (14, 21) add load on the server but not wall-clock time. Loops
that resend a growing transcript (4, 8) cost more than their call count suggests, because each prompt is longer than
the last.

## Suggested reading order

The chapters are numbered by topic, not by difficulty. To build understanding step by step, read them in the stages
below. Each stage relies on ideas from the one before it and ends with a question you should be able to answer
before you move on. Run each sample as you read its chapter, then try at least one of its "Try this" changes.

**Stage 1: what an agent is.** Read [1](#1-calling-a-model--chathelloollama) → [2](#2-native-tool-calling--toolstoolcallingagent) → [4](#4-react-reason-act-observe--graphreactagentgraph) → [3](#3-your-first-graph-evaluator-and-optimizer--graphwriterreviewergraph).
A model call, then a model that calls tools, then the loop made visible (ReAct), then the same loop as a graph with
state and edges. Chapter 4 before 3 shows the loop as plain text first, so the graph in chapter 3 then reads as
the same loop, now drawn as a graph.
*Check:* can you explain why an agent is "an LLM in a loop with tools", and what stops the loop?

**Stage 2: making one call dependable.** Read [10](#10-structured-output--chatstructuredoutputagent) → [13](#13-guardrails--chatguardrailsagent) → [11](#11-conversation-memory--chatconversationmemoryagent) → [12](#12-retrieval-augmented-generation--ragragagent).
Before building bigger workflows, learn to trust a single step: validate what comes back, check what goes in and
out, decide what context the model sees, and ground it in documents.
*Check:* where in a prompt does untrusted text go, and why does code (not the model) validate the reply?

**Stage 3: workflow patterns.** Read [5](#5-prompt-chaining--graphpromptchaininggraph) → [6](#6-routing--graphroutergraph) → [7](#7-orchestrator-and-workers--graphorchestratorworkersgraph) → [8](#8-multi-agent-supervisor--graphsupervisorgraph) → [9](#9-human-in-the-loop--graphhumanapprovalgraph).
These go from fixed steps (chaining) to steps chosen at run time (routing, orchestrator) to an LLM deciding who
acts next (supervisor), and end with a person approving the risky step.
*Check:* for a task you care about, which of these would you pick, and why not a fully autonomous agent?

**Stage 4: reasoning and planning.** Read [14](#14-self-consistency-and-parallel-calls--chatselfconsistencyagent) → [18](#18-plan-and-execute-with-replanning--reasoningplanexecutegraph) → [19](#19-reflexion--reasoningreflexiongraph) → [20](#20-tree-of-thoughts--reasoningtreeofthoughtsagent).
Sampling several answers, planning ahead, learning from a failed attempt, and searching a tree of options. Each
spends more model calls to get a better answer.
*Check:* what does each technique cost in calls, and what kind of problem justifies that cost?

**Stage 5: retrieval and memory in depth.** Read [15](#15-embedding-rag-with-rewriting-and-reranking--ragembeddingragagent) → [16](#16-corrective-rag--ragcorrectiveraggraph) → [17](#17-long-term-memory--memorylongtermmemoryagent).
Embeddings, query rewriting, reranking, a graph that retries retrieval, and memory that outlives the process.
*Check:* why is stored memory an attack surface, and how does the sample limit what gets saved?

**Stage 6: more tools and more agents.** Read [25](#25-tools-from-an-mcp-server--toolsmcpmcpagent) → [26](#26-tool-retrieval--toolstoolretrievalagent) → [27](#27-handoffs-between-agents--multiagenthandoffgraph) → [28](#28-debate--multiagentdebategraph).
Tools from someone else's server, too many tools for one prompt, and agents that pass work to each other or argue.
Compare chapter 27 with chapter 8 (supervisor).
*Check:* why are MCP tool descriptions and results treated as untrusted?

**Stage 7: the framework's power tools.** Read [21](#21-parallel-branches--langgraphparallelbranchesgraph) → [22](#22-subgraphs--langgraphsubgraphgraph) → [23](#23-time-travel-with-checkpoints--langgraphtimetravelgraph) → [24](#24-streaming--langgraphstreamingagent).
LangGraph4j features you reach for once your graphs grow: parallel branches, subgraphs, checkpoints and replay,
and streaming.
*Check:* when do branches really run in parallel, and why does chapter 9 already need a checkpointer?

**Stage 8: running agents for real.** Read [29](#29-evaluating-agents--opsevalharness) → [31](#31-observability-and-token-budgets--opsobservedllm) → [30](#30-model-cascade-and-resilience--opsmodelcascade) → [32](#32-semantic-caching--opssemanticcache).
Measure quality first, then watch cost and latency, then make calls resilient and cheaper. Evaluation comes first
because every other change needs a way to tell whether it helped.
*Check:* how would you know that a prompt change made your agent better and not just different?

**Shorter paths.** If you only have an hour, read stages 1 and 3. If you are building a question-answering bot,
read stage 1, then 10, 12, 13, 15 and 16, then 29. If you are building an autonomous agent with tools, read
stage 1, then 13, 9, 25, 18 and 31. Keep the [Glossary](#glossary) open as you go.

---

## 1. Calling a model — `chat.HelloOllama`

**The idea.** The smallest possible program: build a chat model client, send one prompt, and print the reply.
Everything else in this repository is code around this one call.

**How the sample does it.** `OllamaConfig.chatModel()` builds a LangChain4j `OllamaChatModel` from environment
variables, with a low temperature (0.2), a 10-second connect timeout and a 2-minute reply timeout.
`model.chat(prompt)` returns the reply text.

**Going deeper.**
- *Two phases, two costs.* The server first reads the whole prompt in one parallel pass (*prefill*), then writes
  the reply one token at a time (*decode*). Decoding dominates the wait: a 500-token reply takes far longer than
  reading a 500-token prompt. That is why streaming (24) helps so much, and why budgets reserve room for the reply
  (31).
- *What temperature does.* The model's scores for each candidate token are divided by the temperature before being
  turned into probabilities. Below 1 the favourite becomes more of a favourite; above 1 the choice spreads out. A
  low temperature makes the *most likely* answer more likely. That helps only when the most likely answer is right.
- *Roles are text too.* A chat API's system, user and assistant messages are rendered into one prompt with special
  markers. The model was trained to give the system message weight, but nothing enforces it. This is part of why
  prompt injection works.
- *Cold start.* Ollama loads a model into memory on first use and unloads it after a few idle minutes, so the first
  call after a pause is much slower than the rest. Keep that in mind when you time things.

**What to look for.** The model name and the reply. Run it twice: with temperature 0.2 the answers are similar but
rarely identical.

**Try this.**
- Change the temperature in `OllamaConfig` to 1.0 and compare a few runs.
- Ask the same question to two different models (`OLLAMA_MODEL=...`).

**In production.** Every outbound call needs a timeout; this one has 2 minutes. Give connecting its own short limit
(LangChain4j's `timeout(...)` sets both to one value, so an unreachable host would hold each retry for the full 2
minutes). Never hard-code the URL or the model name; here they come from the environment.

## 2. Native tool calling — `tools.ToolCallingAgent`

**The idea.** Models are bad at arithmetic and know nothing after their training date. *Tools* let the model ask
your code to do something, such as compute, search or look up data, and then use the result. With *native* tool
calling, the model API itself supports it: you send tool descriptions with the prompt, and the model replies with a
structured "call `multiply(12.5, 8)`" request instead of text.

**How the sample does it.** `Calculator` has methods annotated with `@Tool("...")`, and each parameter has a `@P`
description. The model reads those descriptions, so write them for the model. `AiServices.builder(MathAssistant.class)`
generates an implementation of the `MathAssistant` interface. It sends the system message and the tool schemas,
runs any tool calls the model asks for, feeds the results back, and repeats until the model gives a text answer.
`MessageWindowChatMemory.withMaxMessages(20)` keeps the last 20 messages as conversation memory.

**Going deeper.**
- *The model never runs anything.* Tool-calling models are fine-tuned to emit a special structured reply when tools
  are offered. The framework parses that reply, runs your Java method and sends the result back as a new message.
  This is the ReAct loop of chapter 4, with the text format hidden inside the API.
- *Tool schemas are prompt tokens.* Each tool's name, description and parameters are inserted into the prompt on
  every call. The model chooses a tool by *reading* those descriptions, so a description is a prompt. Writing it
  well matters more than the method name. Chapter 26 takes this further and searches over descriptions.
- *Why models need a calculator.* Tokenizers split numbers into chunks that don't line up with digits, and the
  model learned arithmetic as patterns rather than as an algorithm. It gets common sums right and unusual ones
  subtly wrong, which is the worst kind of error. Pushing exact work into code is a theme of the whole guide.
- *The confused deputy.* The model calls tools with *its* authority, but text from users or documents can steer
  which tool and which arguments. Each tool should validate its inputs as if a stranger had typed them.

**What to look for.** The answer to "square root of (12.5 × 8) plus 4" should be 14. `Calculator` prints a
`[tool] ...` line each time the model calls a tool. Turn on `logRequests(true)` in `OllamaConfig` to see the raw
model traffic as well.

**Try this.**
- Add a `@Tool` for powers and ask "what is 2 to the 10th?".
- Make one tool's description vague and watch the model misuse it.

**In production.**
- Not every model supports native tool calling. Gemma, for example, often doesn't, which is why the other samples
  use plain-text protocols (chapter 4).
- Tools that change things (send, delete, pay) need a human check (chapter 9).
- Treat tool arguments as untrusted input, because the model chose them.

## 3. Your first graph: evaluator and optimizer — `graph.WriterReviewerGraph`

**The idea.** One model call writes a draft and another critiques it. The draft is revised until the critic
approves or a revision limit is reached. This *evaluator-optimizer* loop is the simplest way to get better output
than a single call gives.

**How the sample does it.** Two nodes, `writer` and `reviewer`, share a `WriterState`. The reviewer replies with
exactly `APPROVED` or with feedback. The conditional edge after `reviewer` calls `isDone(state)`, which is true when
the feedback starts with `APPROVED` or `MAX_REVISIONS` (2) drafts have been written. If it is false, the graph goes
back to `writer`, whose prompt now includes the feedback.

```
START → writer → reviewer ─┬─ approved / limit → END
           ↑               │
           └──── revise ───┘
```

**Going deeper.**
- *Why a second call to the same model helps.* Writing asks the model to get everything right in one left-to-right
  pass. Reviewing asks a narrower question ("what is wrong with this?") about text the model can now see all at
  once. Same model, different conditioning, different strengths. This is the generator–verifier gap (idea 3) in its
  cheapest form.
- *Its limits.* The reviewer shares the writer's knowledge and blind spots, and models tend to rate text that looks
  like their own output kindly (*self-preference bias*). Most of the gain comes in the first revision. Later ones
  often just churn, or drift away from the original request.
- *Feedback quality decides everything.* "Make it better" produces change, not improvement. Feedback that names a
  specific defect ("the second sentence repeats the first") gives the writer something to act on.
- *The cap turns an open loop into a known cost.* Here the worst case is two drafts and two reviews. Without the
  cap, the worst case is unbounded.

**What to look for.** The log has one `writer: produced draft #n` line per draft and the reviewer's feedback after
each draft, then the final draft. Note that the limit, not the model, guarantees the loop ends.

**Try this.**
- Make the reviewer stricter ("reject anything over 50 words") and watch the revision count.
- Set `MAX_REVISIONS` to 5.

**In production.**
- A model reviewing its own output shares its own blind spots. Where you can, use a code check instead (a test, a
  validator, a word count), as chapters 5 and 10 do.
- Cap the revisions, because each one costs a model call.

## 4. ReAct: reason, act, observe — `graph.ReActAgentGraph`

**The idea.** ReAct (Yao et al., 2022) is the classic agent loop. The model writes a **Thought** about what it
knows, picks an **Action** (a tool and its input), and the code runs the tool and appends an **Observation**. This
repeats until the model writes a **Final Answer**. It works with any instruction-following model because the
"protocol" is plain text, so native tool calling isn't needed.

**How the sample does it.**
- The `reason` node sends the question, the tool list and the whole transcript so far.
- The reply is parsed with regexes for `Thought:`, `Action: tool[input]` and `Final Answer:`.
- The `act` node looks up the tool in `ReActTools` and appends `Observation: <result>`. The tools are `lookup` (a
  small fact table) and `calculate` (one `+ - * /` operation in `BigDecimal`; division is rounded).
- Two defences against real model behaviour:
  - `truncateAtObservation` discards any Observation the model invents for itself.
  - An unparseable reply produces an "Invalid action..." observation that tells the model the format again.
- `MAX_STEPS` (6) caps the loop.

```
START → reason ─┬─ Final Answer / step limit → END
           ↑    └─ Action → act
           └──── Observation ──┘
```

**Going deeper.**
- *Why interleave thinking and acting.* Reasoning alone (chain of thought) works from memory and can build a chain
  on an invented fact. Acting alone has no plan. Interleaving grounds each next thought in a real observation, so
  the environment corrects the model's mistakes as it goes.
- *The transcript is the agent's whole state.* There is no hidden memory: what the model saw at each step is
  exactly the prompt. Save it and you can replay any decision. That property is worth keeping in any agent you build.
- *Why the model invents observations.* It has seen many Thought/Action/Observation transcripts, and after an
  Action the most likely next text is an Observation, so it writes one. `truncateAtObservation` cuts it off. Real
  frameworks pass a *stop sequence* (`"Observation:"`) so the server stops generating at that point.
- *Cost grows quadratically.* Step *k* resends the *k − 1* steps before it, so a run of *n* steps sends roughly
  *n²/2* steps' worth of prompt. Doubling the step limit roughly quadruples the prompt tokens. This is why
  `MAX_STEPS` matters for cost and not only for termination, and why chapter 31 adds a token budget.

**What to look for.** The printed transcript is the Thought/Action/Observation part of what the model saw on each
step; the prompt adds the instructions, the tool list and the question above it. This is the best place in the
repository to build an intuition for how agents "think".

**Try this.**
- Add a tool, such as a unit converter, to `ReActTools.defaults()` and ask a question that needs it.
- Compare this with chapter 2: the same idea, with the protocol visible instead of hidden inside the framework.

**In production.** Text protocols break when a model formats a reply slightly differently, so parse leniently and
answer bad output with a corrective message. Keep the transcript bounded, because long loops overflow the context
window.

---

## 5. Prompt chaining — `graph.PromptChainingGraph`

**The idea.** Split a task into a *fixed* sequence of small steps, each working on the previous step's output:
outline → draft → polish. Small prompts are easier for a model to follow than one big prompt. Between steps, a
*gate* written in plain code checks the intermediate result before passing it on.

**How the sample does it.**
- The `outline` node asks for 3–5 bullet points.
- The conditional edge `gate(state)` counts list items in code with `countPoints`, which accepts `-`, `*`, `•`, `1.`
  and `1)`.
- With 3 or more points, the graph continues to `draft` and then `polish`. Otherwise it retries the outline, and the
  retry prompt says what was wrong ("only 1 point(s)"). After `MAX_OUTLINE_ATTEMPTS` (2) it gives up instead of
  drafting from a bad outline.

```
START → outline → gate ─┬─ valid → draft → polish → END
           ↑            ├─ invalid, retry ┘
           └────────────┘
                        └─ invalid, limit → END
```

**Going deeper.**
- *Why small steps beat one big prompt.* Each call has one goal, so instructions don't compete for the model's
  attention. Each intermediate result can be inspected and checked. Each step can also use its own temperature or
  model: low for the outline, higher for the prose.
- *What it costs.* Latency adds up, because the steps run one after another. Information can also get lost: a later
  step sees only what you pass it. If `polish` needs the original request, it has to be in `polish`'s prompt.
- *Gates catch errors where they are cheapest.* A bad outline caught by the gate costs one retry. Caught by a human
  after `polish`, it costs the whole run. This is idea 4, compounding errors, applied directly.
- *Why the retry prompt says what was wrong.* At low temperature the same prompt gives nearly the same reply, so a
  blind retry mostly reproduces the failure. Adding the error changes the input, which is the only thing that
  changes the output.

**What to look for.** The log lines `outline: attempt 1, 3 point(s)`, `draft: written`, `polish: done`.

**Try this.**
- Make the placeholder return a single-line outline on the first call and a good one on the second (for example with
  an `AtomicInteger`), and watch the retry with feedback. A placeholder that always returns a bad outline shows the
  give-up path instead.
- Add a `fact_check` step between `draft` and `polish`.

**In production.**
- Use gates in code wherever the check can be expressed in code: length, format, required sections, a JSON schema.
  They are free, fast and deterministic.
- Retrying with the identical prompt usually gets the identical answer, so say what was wrong.

## 6. Routing — `graph.RouterGraph`

**The idea.** Classify the input first, then send it to a specialist with its own focused prompt (or its own model,
or its own tools). Routing beats one do-everything prompt because each specialist prompt can be short and precise.

**How the sample does it.**
- `classify` asks for one word: billing, technical or general.
- `normalizeRoute` takes the model's reply and returns a label only when the reply names *exactly one* known route.
  A reply that names none, or several ("technical, not billing"), goes to `general`.
- Each specialist node prepends its own instructions from `SPECIALISTS`.

```
                  ┌─ billing ──┐
START → classify ─┼─ technical ┼→ END
                  └─ general ──┘
```

**Going deeper.**
- *Classification is the easiest job you can give a model.* The answer is one label from a short, closed list.
  Small, cheap models do it well, and that is where routing saves the most money: classify with a small model, then
  spend a big one only where it is needed. A classic text classifier or an embedding nearest-neighbour lookup can
  also do this job deterministically with no generation at all.
- *Why specialists win.* A do-everything prompt grows long and its rules start to conflict ("be brief", "explain
  in detail"). A specialist prompt is short and coherent. Each route can also get only *its* tools, which is least
  privilege per route: a billing question can never reach a tool that changes server settings.
- *Why "exactly one label".* A naive `contains` check on "technical, not billing" matches whichever label it tests
  first. Requiring exactly one known label turns ambiguity into the default route instead of a coin toss.
- *Choose the default for when it's wrong.* Misroutes are certain to happen, so make the default the route that does
  the least harm when it's wrong. A general assistant that can answer anything a little is a safer default than any
  specialist.

**What to look for.** Three demo queries, each printed with its route and its answer.

**Try this.**
- Add a `sales` route. It needs three edits: a `SPECIALISTS` entry (which adds the node and teaches `normalizeRoute`
  the label), the category list in the classifier prompt, and the conditional-edge map. A good refactor is to build
  the last two from `SPECIALISTS.keySet()`, so one entry is enough.
- Route by cost instead: easy questions to a small model, hard ones to a big one.

**In production.**
- Never let raw model output pick a code path. Map it onto an allow-list and have a default.
- Log the chosen route; misroutes are the most common bug in routed systems.

## 7. Orchestrator and workers — `graph.OrchestratorWorkersGraph`

**The idea.** Unlike prompt chaining, the steps aren't known in advance. An *orchestrator* call breaks the task into
subtasks **at run time**, a *worker* handles each subtask with only the context it needs, and a *synthesizer* merges
the results.

**How the sample does it.**
- The orchestrator's reply is parsed by `parsePlan`, which strips list markers, drops preamble lines such as "Here
  are the subtasks:", and keeps at most `MAX_SUBTASKS` (4).
- `NEXT_INDEX` in the state points at the next subtask. The `worker` node handles one subtask and advances the index.
- The conditional edge loops back to `worker` until all subtasks are done, then goes to `synthesizer`.
- `RESULTS` uses `appenderWithDuplicate`, so two workers that return the same text don't leave the results shorter
  than the plan.

```
START → orchestrator → worker ─┬─ subtasks left → worker
                                └─ all done → synthesizer → END
```

**Going deeper.**
- *Open loop versus closed loop.* The orchestrator plans once and never looks back. That is *open-loop* control,
  like a toaster timer. Chapter 18 adds feedback: it replans after each step, which is *closed-loop* control, like a
  thermostat. Open loop is cheaper and predictable. Closed loop copes with surprises.
- *The plan bounds the result.* Workers can't fix a bad decomposition. Overlapping subtasks produce duplicated
  work that the synthesizer then has to reconcile. A missing subtask leaves a hole nobody notices. Most of the
  quality comes from the planning prompt.
- *Why isolated workers.* Each worker sees only its subtask. Prompts stay small, workers can't distract each other,
  and they could run in parallel (21) or on different models.
- *The cost is easy to predict.* One plan, *n* workers and one synthesis: *n + 2* calls, with *n* capped at
  `MAX_SUBTASKS`.

**What to look for.** The numbered list "subtask → result", then the combined report.

**Try this.**
- Make the placeholder plan 7 subtasks and watch the cap keep 4.
- The workers here run one after another. Real systems often run them in parallel because they are independent.

**In production.**
- Cap the plan size, because a verbose model can plan 30 subtasks and spend 30 calls.
- Give workers only the context they need; smaller prompts are cheaper and less confused.

## 8. Multi-agent supervisor — `graph.SupervisorGraph`

**The idea.** Several specialist "agents" (here, a researcher, a writer and a critic, each defined by a persona
prompt) share a scratchpad. A *supervisor* reads the scratchpad and decides who acts next, or says `FINISH`. Control
always returns to the supervisor, so the order is decided at run time, not fixed in code.

**How the sample does it.**
- `AGENTS` is an ordered list of `Agent(name, persona)` records. The graph adds one node per agent, each with an edge
  back to `supervisor`.
- The supervisor replies "NEXT: writer" or "FINISH". `parseNext` takes the first known agent name after `NEXT:`, so
  "NEXT: the writer" and "**NEXT:** writer" both work. Anything else means finish.
- `MAX_TURNS` (6) stops a supervisor that never finishes.
- `WORK` keeps every contribution as "agent: output", duplicates included.

**Going deeper.**
- *A supervisor is routing in a loop.* Each supervisor turn is chapter 6 again: classify the situation and choose a
  specialist. The difference is that the input is a growing scratchpad, and the route can be "done".
- *The "agents" are one model wearing different hats.* Each one is the same LLM with a different persona prompt.
  Multi-agent design is mostly about *separating instructions and context*, not about different minds. That's why
  one agent with tools is often enough.
- *The shared scratchpad has a price.* Every agent sees everything, which helps collaboration but makes prompts
  grow every turn. It also spreads mistakes and injected text: one agent's bad output reaches all the others.
  The supervisor's choices tend to get worse as the scratchpad grows.
- *Count the calls.* Each agent turn costs two calls, one for the supervisor and one for the agent. A handoff design
  (27) saves the supervisor's call at the cost of losing the global view.

**What to look for.** The scratchpad: researcher facts, the writer's draft, the critic's "LGTM". Then the turn
count (4) and the answer, which is the writer's latest draft.

**Try this.**
- Add a `fact_checker` agent: one list entry.
- Make the supervisor never say `FINISH`: the placeholder finishes as soon as the critic has spoken, so change the
  supervisor rules, not just the critic. Then watch `MAX_TURNS` end the run.

**In production.**
- Multi-agent systems are expensive and hard to debug. Use them when the roles really need different instructions
  or tools; otherwise one agent with tools is simpler.
- Always cap the turns, and keep the agent names on an allow-list.

## 9. Human in the loop — `graph.HumanApprovalGraph`

**The idea.** Before an agent does something irreversible or outward-facing, such as sending an email, spending
money or deleting data, *pause* and let a person approve or reject the action. The run must be able to stop, wait
indefinitely, and resume exactly where it left off.

**How the sample does it.** Two LangGraph4j features:
- a **checkpointer** (`MemorySaver`), which saves the state after every node, keyed by a *thread id*;
- `interruptBefore("human_review")`, which stops the run just before that node.

`run(...)` invokes the graph, which stops after `propose`. It then shows the proposal to a reviewer, records the
decision with `graph.updateState(config, Map.of(APPROVED, true/false))`, and resumes with `GraphInput.resume()`. The
edge after `human_review` goes to `execute` only if approved. Missing state means "not approved", so a forgotten
decision never triggers the action. `execute` runs `performAction` in code, and its result is what really happened.
The LLM is never asked to *say* the action succeeded.

```
START → propose → ⏸ human_review ─┬─ approved → execute → END
                                  └─ rejected → cancel ──→ END
```

**Going deeper.**
- *Approve before, not after.* Approval after an action is an audit, not a control. The interrupt has to sit
  *before* the node with the side effect.
- *Pausing makes the run durable.* The checkpointer saves the whole state at the pause, so in principle the process
  could exit and a different one could resume tomorrow. Time travel (23) uses the same mechanism.
- *Bind the approval to what was shown.* If the proposal can change between "shown" and "executed", the approval
  covers something the person never saw. This is a *time-of-check to time-of-use* bug. A hash of the proposal,
  stored with the approval and checked before `execute`, closes it.
- *Approval fatigue.* If every step asks for approval, people stop reading and click yes. Gate only the irreversible
  or outward-facing actions, and show exactly what will happen ("send this email to these 3 people"), not the
  model's reasoning.

**What to look for.** "Approve? [y/N]". Answer `y`, `n`, or press Ctrl-D, which counts as a rejection.

**Try this.**
- Let the reviewer edit the proposal before approving: update `PROPOSAL` with `updateState`.
- Swap `MemorySaver` for a persistent checkpointer, so the approval can come from a web page hours later.

**In production.**
- Bind the approval to the exact proposal that was shown.
- Default to "no".
- Log who approved what.
- Never let model text stand in for the result of an action.

---

## 10. Structured output — `chat.StructuredOutputAgent`

**The idea.** Code needs data, not prose, so ask the model for JSON matching a schema, parse it into a typed
object, and *validate* it. When validation fails, send the exact error back and let the model repair its answer.

**How the sample does it.**
- The prompt gives the schema for `ReviewSummary(product, rating 1–5, sentiment)`.
- `stripFences` keeps only the outermost `{...}`, which drops the Markdown fences models love to add and any prose
  around the JSON.
- `parse` uses a strict Jackson mapper (no float-to-int, no string-to-number coercion), so `4.9` or `"5"` fails
  instead of slipping through, and then checks each field's range.
- On failure, the repair prompt repeats the **whole task**: the model is stateless, so without the original review
  it would have nothing to fix from. It also adds the error and the bad reply.
- `MAX_ATTEMPTS` is 3.

**Going deeper.**
- *Why models break JSON.* The model writes one token at a time with no parser checking it. Its training is full of
  chat answers that wrap code in fences and add a friendly sentence. Long, nested output gives it more chances to
  slip on a quote or a comma.
- *Constrained decoding.* Ollama's JSON-schema mode masks, at every step, the tokens that would make the output
  invalid. The output is then well-formed, unless the token limit cuts the reply off mid-object. It still can't make the *values* right. Forcing the format from
  the first token can also cut short the model's reasoning, which is why some designs let it think in prose first
  and emit JSON last.
- *Why strict parsing.* Lenient coercion (`"5"` → 5, 4.9 → 4) hides the fact that the model was confused. Failing
  loudly and asking for a repair surfaces it, and the precise error message gives the repair a target.
- *The schema is a contract.* Everything after the parser can trust the record's types and ranges. Validation is the
  boundary where untrusted model text becomes trusted data.

**What to look for.** Attempt 1 has `"rating": 8`. Attempt 2 is the corrected JSON, followed by the parsed record.

**Try this.**
- Add a `pros` list field.
- Feed it a review in another language.

**In production.**
- Ollama can constrain the output to a JSON schema, and LangChain4j AI Services can return records directly. Use
  them when your model supports them, and still validate the values, because valid JSON can hold wrong values.

## 11. Conversation memory — `chat.ConversationMemoryAgent`

**The idea.** A model remembers nothing between calls; "memory" is what you resend in the prompt. Resending the whole
conversation eventually overflows the context window. So keep the recent turns word for word and compress older ones
into a running summary.

**How the sample does it.**
- Up to `RECENT_TURNS` (4) recent messages are kept verbatim.
- When a fifth arrives, the two oldest are folded into the summary by an LLM call ("keep every fact the user
  stated"), and the summary is capped in code at `MAX_SUMMARY_CHARS` (600).
- Each prompt is: instructions, summary, recent turns, then the new message.

**Going deeper.**
- *A three-way trade-off.* Fidelity, cost and conversation length can't all be maximised. Keeping everything
  verbatim is faithful but grows without bound. Summarising is bounded but lossy.
- *Summaries lose the wrong things.* The summarising call decides what matters *now*, without knowing what will
  matter later. Exact names, numbers and dates are the first things to blur. Summaries of summaries drift further
  with each pass. That's why the prompt says "keep every fact the user stated".
- *Alternatives.* Instead of compressing the past, you can store it and *retrieve* the relevant turns for each new
  message. This is RAG over the conversation, the idea behind chapter 17. Real assistants often combine a window,
  a summary and retrieval.
- *Stable prefixes are cheaper.* Many model servers cache the work done on a prompt's beginning. A prompt that
  starts with the same instructions every time lets that cache be reused, while one that rewrites its opening each
  turn throws it away.

**What to look for.** By the fifth message, "I'm Priya" has left the recent turns and only the summary still holds
it. The final answer works only because the summary kept the name.

**Try this.**
- Set `RECENT_TURNS` to 2.
- Make the summary prompt sloppy ("summarise briefly") and see which facts disappear.

**In production.**
- LangChain4j ships `MessageWindowChatMemory` and `TokenWindowChatMemory`. Count tokens, not messages.
- Summaries lose details, so keep facts that must survive in structured memory instead.

## 12. Retrieval-augmented generation — `rag.RagAgent`

**The idea.** Don't trust what the model memorised: retrieve relevant documents first, put them in the prompt, and
tell the model to answer *only* from them and cite them. Answers become checkable, and the knowledge can be updated
without retraining.

**How the sample does it.**
- `KeywordRetriever` ranks a small handbook by TF-IDF cosine similarity: words that are rare across documents count
  more.
- The top `TOP_K` (2) documents go into `<doc id="...">` tags, their text escaped with `PromptText.escape` so a
  document can't close its tag, with instructions to cite `[doc-id]` and a fixed refusal sentence.
- If retrieval finds nothing, the agent answers "I don't know" **without calling the model**, the cheapest defence
  against a confident made-up answer.

**Going deeper.**
- *Why grounding works.* Models are much better at reading comprehension (answering from text in front of them)
  than at recall (answering from training). RAG turns a recall problem into a reading problem. Knowledge moves
  from the weights, which are fixed and uncitable, into the prompt, which is fresh and checkable.
- *What TF-IDF measures.* A word's weight is how often it appears in this document (*term frequency*) times how rare
  it is across all documents (*inverse document frequency*). `KeywordRetriever` uses a smoothed form,
  log(1 + total documents ÷ documents containing the word), so a word found in every document counts almost
  nothing, while "laptops", found in one handbook document, counts a lot.
- *Three ways RAG fails.* The right document isn't retrieved, so the model refuses or makes something up. The wrong
  one is retrieved, so the answer is confidently wrong *and comes with a citation*. Or the right document conflicts
  with what the model learned in training, and the model sides with its training. Only the first is visible
  without reading the answer against its sources.
- *"Answer only from the documents" is a request, not a guarantee.* The model can still blend in what it already
  knows. Citations make that checkable: an answer whose cited document doesn't say what the answer claims has
  failed.

**What to look for.** "Retrieved: [leave-policy (0.56), ...]" and the citation in the answer. The wifi-password
question retrieves nothing and is refused.

**Try this.**
- Add a document and ask about it.
- Ask "How much vacation am I entitled to?". It shares no words with the handbook, so nothing is retrieved, even
  though the leave policy answers it. Keyword search misses synonyms ("vacation" vs "leave"), which is why real
  systems use embeddings. Stop words matter too: "How much holiday do I get?" *does* retrieve the leave policy,
  but only because "get" isn't a stop word and also appears in it.

**In production.**
- Retrieval quality limits answer quality.
- Treat retrieved text as untrusted data, since documents can contain injected instructions.

## 13. Guardrails — `chat.GuardrailsAgent`

**The idea.** Checks in plain code, before and after the model. *Input* guardrails reject a request before it costs
a call. *Output* guardrails inspect the reply before the user sees it.

**How the sample does it.**
- **Input checks:** a length cap (`MAX_INPUT_CHARS`, 1,000), prompt-injection patterns and blocked topics. The text
  is normalised first (Unicode NFKC, invisible characters removed, whitespace collapsed), so "ig​nore  previous"
  is still caught.
- **Fencing:** user text goes inside `<user>` tags, labelled as data, with `<`, `>` and `&` escaped, so `</user >`
  cannot close the tag.
- **Output checks:** email addresses and phone numbers (9 or more digits, so dates survive) are redacted. A reply
  containing the system prompt's marker is blocked.

**Going deeper.**
- *Why patterns can't win.* Language is open-ended. Any blocklist has a paraphrase, a translation, an encoding
  (base64, leetspeak) or a split across two messages that slips past it. And the model reads instructions and data
  through one channel, so no amount of input filtering creates a hard boundary between them.
- *What guardrails are good for.* They cheaply stop obvious abuse and accidents, and they are one layer of
  *defence in depth*. Output checks are the sturdier layer, because they look at what actually leaves the system,
  however the model was tricked into producing it.
- *Why normalise first.* Attackers use look-alike Unicode characters and invisible zero-width characters to break up
  keywords. NFKC normalisation folds compatibility look-alikes (fullwidth letters, ligatures) into plain ones, and
  stripping invisible characters rejoins the words, so the patterns see closer to what a human reader sees.
  Look-alikes from other scripts, such as Cyrillic "а" for Latin "a", need a separate confusables check.
- *Two error rates.* Every guardrail blocks some good requests (false positives) and lets some bad ones through
  (false negatives). Tightening one loosens the other. Measure both on real traffic, because a guardrail that blocks
  legitimate users is a bug too.

**What to look for.** Four requests: allowed, allowed with redactions, blocked as an injection, blocked as a topic.
Each request after the first shows the rules that fired.

**Try this.**
- Find a rephrasing that gets past the injection patterns. You will, which is the point of the next paragraph.

**In production.**
- Pattern checks are a first layer, not a defence. The real defences are limiting what the agent *can* do (least
  privilege, human approval) and checking what it *did*.
- LangChain4j has an `InputGuardrail` / `OutputGuardrail` API for AI Services.

## 14. Self-consistency and parallel calls — `chat.SelfConsistencyAgent`

**The idea.** Ask the same question several times and take a majority vote (Wang et al., 2022). One sample can
reason its way to a wrong answer, but independent samples rarely all make the same mistake. The agreement ratio
doubles as a cheap confidence score.

**How the sample does it.**
- `SAMPLES` (5) calls run concurrently on virtual threads, one per call, which suits calls that spend their time
  waiting on the network.
- Each reply's last `ANSWER:` line is normalised (`*`, `_` and backticks stripped, case folded, a trailing `.` or `!` removed), and
  `vote` picks the most common answer, breaking ties alphabetically.
- A deadline (`SAMPLE_TIMEOUT`, 3 minutes for all samples together) bounds the wait. A sample that fails or is late
  is cancelled and *abstains* instead of sinking the whole vote.

**Going deeper.**
- *The maths of voting.* If each sample is right with probability 0.6 and the samples are independent, a majority
  of 5 is right about 68% of the time, and more samples push that higher (the *Condorcet jury theorem*). In practice
  it does even better, because wrong answers tend to scatter across many different values while right answers agree.
- *The catch is independence.* The samples come from the same model with the same prompt. A shared misconception
  produces the *same* wrong answer every time, and voting turns that into a confident, unanimous wrong answer.
  Voting removes random slips, not systematic errors.
- *Temperature is the diversity knob.* At temperature 0 all samples are identical and you pay N times for one
  answer. Too high, and samples become noise. Moderate temperatures give real variation in reasoning paths.
- *It needs comparable answers.* Voting works for short answers that can be normalised (a number, a label, a name).
  Free-form text needs a similarity measure or a judge to group answers first.
- *Agreement is a better confidence signal than asking.* A model asked "how sure are you?" is poorly calibrated
  (see chapter 30). The fraction of samples that agree is measured, not claimed.

**What to look for.** "Answer: 4 (4 of 5 samples agree, 80%)". The placeholder deliberately makes one sample slip.

**Try this.**
- With a real model, compare temperature 0.2, where votes barely differ, with 0.8.

**In production.**
- This costs N times the tokens; use it for high-value answers, or when agreement can decide whether to escalate to
  a human.
- Cap concurrency to what your model server can handle.

---

## 15. Embedding RAG with rewriting and reranking — `rag.EmbeddingRagAgent`

**The idea.** Chapter 12 matched words. Real RAG turns text into *embeddings*, vectors where similar meanings sit
close together, and searches by vector distance. Two extra steps make it work in practice:
- **Query rewriting.** Users ask chatty questions ("hey, quick one..."), so the model first rewrites the question as
  a short search query.
- **Reranking.** Vector search is fast but rough, so fetch a few candidates and let the model score each one against
  the original question.

**How the sample does it.**
- `VectorRetriever` uses LangChain4j's own pieces: `DocumentSplitters.recursive(300, 40)` cuts documents into
  overlapping chunks, each chunk is embedded, and the vectors go into an `InMemoryEmbeddingStore`. Each chunk keeps
  its document id as `source`, for citations.
- `ask` rewrites the question, falling back to the question itself if the rewrite comes back blank.
- It searches for `CANDIDATES` (4) chunks with a score of at least `MIN_VECTOR_SCORE` (0.6).
- The model rates each chunk from 0 to 10. `parseScore` understands "8/10", "9 out of 10" and scale-first replies
  such as "On a 0-10 scale: 9".
- It keeps the best `TOP_K` (2) chunks that score at least `MIN_RERANK_SCORE` (6). If none survive, it says "I don't
  know" without an answer call.
- Chunks go into the rerank and answer prompts escaped (`PromptText.escape`) and labelled as data, so a document
  containing `</doc>` can't break out of its fence.

**Watch the score scale.** LangChain4j store scores are *relevance* scores, `(cosine + 1) / 2`. A score of 0.5 means
unrelated, not "half relevant", so useful thresholds sit above 0.5.

**Going deeper.**
- *What an embedding model learned.* It was trained on pairs of related texts (a question and its answer, a title and
  its article) to put related texts close together and unrelated ones far apart. Cosine similarity measures the
  angle between two vectors, so it compares direction (topic) rather than length.
- *Why two stages.* Vector search is a *bi-encoder*: question and chunk are embedded separately, so chunk vectors can
  be computed once and searched quickly. A reranker is a *cross-encoder*: it reads question and chunk together,
  which is far more accurate but costs a call per pair. Fetch broadly with the fast stage, then judge carefully with
  the slow one.
- *Where embeddings fail.* Exact identifiers such as error codes, product names and version numbers carry little
  meaning, so embeddings blur them. Keyword search catches them. Production systems usually run both (*hybrid
  search*) and merge the rankings.
- *Chunking is a precision trade.* Small chunks match precisely but lose surrounding context. Large chunks keep
  context but dilute the match. Overlap keeps a fact that straddles a boundary from being cut in half.
- *Why rewrite the query.* Questions and documents are written in different styles. Rewriting moves the query toward
  the documents' language. *HyDE* (Gao et al., 2022) goes further and embeds a hypothetical answer instead of the
  question.
- *One embedding model per index.* Query and documents must be embedded by the same model. Switching models means
  re-embedding every document.

**What to look for.** For each question: the rewritten query, the chunks kept (each with its vector and rerank
scores), and the answer. The leave question keeps `leave-policy (vector 0.88, rerank 9)`. The football question keeps
nothing and gets "I don't know".

**Try this.**
- Compare with chapter 12: ask a question in words the handbook doesn't use. With the placeholder embeddings
  nothing matches; with a real embedding model it often does.
- Lower `MIN_RERANK_SCORE` to 2 (the placeholder rates irrelevant chunks 2) and watch off-topic chunks reach the
  answer, or try it with a real model.

**In production.**
- Tune the thresholds per embedding model; they don't transfer.
- Reranking costs one model call per candidate, so dedicated reranker models are often used instead.
- Swap `InMemoryEmbeddingStore` for a persistent store (pgvector, Qdrant, Chroma); the rest of the code stays.

## 16. Corrective RAG — `rag.CorrectiveRagGraph`

**The idea.** Corrective RAG (Yan et al., 2024) doesn't trust the first search. A grader checks each retrieved chunk.
If none help, the agent rewrites the query and searches again, and only then answers or admits it doesn't know.
This catches the common failure where the user's words ("PTO") don't match the documents' words ("leave").

**How the sample does it.**
- `retrieve` searches for `CANDIDATES` (3) chunks.
- `grade` asks a yes/no question per chunk; yes/no is easier for a small model than a score. `isYes` needs an
  explicit "yes", so an unclear reply means "not relevant". The chunk is fenced and escaped there too, since the
  grade decides what reaches the answer.
- `afterGrading` routes to `generate` if any chunk passed, to `rewrite` while fewer than `MAX_REWRITES` (2) rewrites
  have been made, and to `give_up` otherwise.
- `rewrite` asks for plainer or more formal words, falling back to the question if the reply is blank.

```
START → retrieve → grade ─┬─ some relevant ────────→ generate → END
           ↑              ├─ none, rewrites left ─→ rewrite ─┐
           └──────────────┼──────────────────────────────────┘
                          └─ none, no rewrites left → give_up → END
```

**Going deeper.**
- *Retrieval as a feedback loop.* Plain RAG is open loop (chapter 7's term): it retrieves once and trusts the result.
  CRAG closes the loop by checking the retrieval and acting on the check. Most of the gain comes from *noticing* bad
  retrieval, not from any clever retry.
- *Small decisions beat big ones.* Grading splits "did we find relevant text?" from "answer the question". Each is a
  simpler judgement, and a yes/no answer is more reliable from a small model than a score.
- *Grader strictness is a precision/recall trade.* A strict grader rejects useful chunks and causes needless
  rewrites and give-ups. A lenient one lets noise into the answer. Tune it with the evaluation harness (29), not by
  feel.
- *Rewriting can't create knowledge.* If the documents don't contain the answer, no query finds it. The rewrite
  limit and the honest `give_up` path matter as much as the rewrite itself.
- *The next step is agentic RAG.* The model decides *whether* to retrieve, *what* to search for and *when it has
  enough*, using retrieval as a tool in a ReAct loop (4).

**What to look for.** The step log. "Does my PTO roll forward?" retrieves 0 chunks, is rewritten to "unused paid
leave days carry over next year", retrieves 3, 1 of which is graded relevant, and is answered with a citation. "What's
the dress code?" is rewritten twice, never finds anything, and gives up.

**Try this.**
- Set `MAX_REWRITES` to 0 and compare.
- Make the grader stricter ("only if it fully answers the question").

**In production.**
- Grading costs one call per chunk, so grade a few candidates, not dozens.
- The full CRAG paper also falls back to web search when the documents fail; the `give_up` node is where that would go.

## 17. Long-term memory — `memory.LongTermMemoryAgent`

**The idea.** Chapter 11 remembered within one conversation. *Long-term* memory survives restarts: the agent pulls
durable facts out of what the user says ("follows a vegetarian diet"), saves them, and in a later session recalls the
facts that relate to the new message. This is how assistants "remember you".

**How the sample does it.**
- After each user message, an extraction call returns `FACT: ...` lines. The facts are stored in a `FactStore`, a
  JSON file of fact texts. Embeddings are recomputed in one batch on load.
- Each answer recalls up to `RECALL_K` (3) facts whose cosine similarity to the message is at least
  `MIN_RECALL_SCORE` (0.1). They go into a `<memory>` block, escaped and labelled as data.
- `FactStore` keeps memory bounded and safe to write: at most `MAX_FACTS` (200) facts of at most `MAX_FACT_CHARS`
  (200) characters, duplicates removed, the newest kept. Writes go to a temp file that is then moved into place.
- `main` runs two "sessions" with two separate agent objects over the same file, to show recall after a restart.

**Memory is an attack surface.** Whatever is stored is injected into every later prompt; this is *memory poisoning*
(OWASP ASI06). So:
- facts are extracted from the **user's** message only, never from the model's reply;
- a message that looks like it holds a secret stores **nothing**. The check covers a credential keyword, a card
  number that passes the Luhn check, and an API-key prefix (`AKIA`, `sk-`, `ghp_`);
- the user can erase facts with "forget <phrase>", which runs without a model call;
- the demo keeps its file in a private temp directory, and `AGENT_MEMORY_FILE` keeps memory between runs.

**Going deeper.**
- *Three kinds of memory.* Agent research borrows from cognitive science (for example CoALA, Sumers et al., 2023):
  *episodic* memory holds what happened, *semantic* memory holds facts ("is vegetarian"), and *procedural* memory
  holds how to act (in CoALA, the agent's code and the model's weights). This sample keeps semantic memory.
- *Writing is the hard part.* Deciding what is worth keeping, noticing when a new fact contradicts an old one, and
  letting stale facts expire are all harder than recall. Most memory bugs are write bugs.
- *Similar is not the same as relevant.* Recall finds facts whose *wording* is close to the message. "What should I
  cook tonight?" needs the diet fact, but the two share few words, and only a good embedding model bridges that.
  Some systems ask the model which memories matter instead of trusting similarity alone.
- *Poisoning persists.* A prompt injection in one conversation affects one answer. An injection stored as memory
  affects every later session. That is why the sample filters on the *write* side, which is where the risk enters.

**What to look for.**
- Session 1 stores two facts, refuses the password message, and forgets "learning Java agents".
- Session 2 recalls the diet and suggests a vegetarian dinner.

With the placeholder embeddings, recall needs a shared word ("diet"); a real embedding model also links "dinner" to
"vegetarian".

**Try this.**
- Set `AGENT_MEMORY_FILE=/tmp/me.json`, run it, edit the messages in `main`, and run it again: the second run
  starts with the first run's facts.
- Write a fact that contradicts an old one ("I eat fish now"). Both are kept. Real systems need an *update* step that
  replaces outdated facts.

**In production.**
- Decide what may be remembered, for how long, and how users can see and delete it. Memory is personal data.
- Pattern-based secret detection is a first layer, so keep secrets out of conversations entirely where you can.

---

## 18. Plan-and-execute with replanning — `reasoning.PlanExecuteGraph`

**The idea.** Chapter 7 planned once and then executed blindly. In the real world a step often reveals something
that changes the plan: you call the restaurant and it's closed. *Plan-and-execute with replanning* writes a plan,
does **one** step, and then asks the model to rewrite the *remaining* plan in light of what it learned, or to
declare the goal met.

**How the sample does it.**
- `planner` writes a numbered plan, parsed by `TextLists.items` and capped at `MAX_PLAN_STEPS` (5).
- `executor` does only the first step and records "step → result" in `PAST_STEPS`.
- `replanner` sees the goal, everything done so far and the old remaining steps. It replies either `FINAL: <answer>`
  or a new numbered list of remaining steps. Each plan version is kept in `PLAN_HISTORY`, so you can see how the plan
  changed.
- `afterReplan` routes on: an answer goes to END; otherwise more steps go back to `executor`; and an empty plan, or
  `MAX_STEPS` (6) executed, goes to `finish`, which always writes a summary answer, so a run never ends with nothing.

```
START → planner → executor → replanner ─┬─ FINAL: answer → END
                     ↑                  ├─ new plan → executor
                     └──────────────────┘
                                        └─ plan empty / step limit → finish → END
```

**Going deeper.**
- *Why plan at all.* ReAct (4) decides one step at a time and never looks ahead. An explicit plan gives lookahead,
  something a person can review *before* anything runs, and a commitment the next steps can be held to.
- *Replanning is a feedback controller.* After each step the replanner compares where things are with where they
  should be and corrects course. It costs a call per step, which is chapter 7's open-loop/closed-loop trade made
  concrete.
- *A cheaper variant.* Replan only when a step's result is surprising, and otherwise follow the existing plan. That
  saves most of the replanning calls when the plan is good.
- *Plan drift.* Over several rewrites a model can quietly drop part of the goal. Repeating the original goal in
  every replan prompt, as this sample does, anchors it.

**What to look for.**
- Plan v1 books "Steak Corner".
- The first step finds that two people are vegan.
- Plan v2 switches to "Green Fork". Two steps later the replanner says `FINAL:`.

**Try this.**
- Make the booking step fail ("Green Fork is full") and write a replanner rule that recovers.
- Compare the number of model calls with chapter 7: replanning costs one extra call per step.

**In production.**
- Replanning is what makes long tasks robust, but each step now costs two calls (execute and replan). Cap the steps.
- Keep the past-steps list compact, because it grows with every step.

## 19. Reflexion — `reasoning.ReflexionGraph`

**The idea.** Reflexion (Shinn et al., 2023) is learning from failure, *in words*. After a failed attempt the model
writes a short lesson ("15 is not prime, check for divisors"), and every later attempt is shown all the lessons so
far. The key ingredient is an **objective judge**, here a checker in code, so the agent learns from facts rather
than from its own opinion.

**How the sample does it.**
- The task: list three primes above 10, in ascending order.
- `checkPrimes` checks, in plain Java: exactly three whole numbers, each above 10, each prime, strictly ascending. It
  returns a precise error message.
- `attempt` lists every lesson so far in the prompt. `check` runs the checker and logs the verdict. On failure,
  `reflect` turns the error into one general lesson.
- `MAX_ATTEMPTS` (3) caps the loop.
- `run(..., priorLessons)` starts a new task with lessons from earlier tasks. This is the "memory" half of Reflexion:
  lessons outlive the task.

```
START → attempt → check ─┬─ passed / attempt limit → END
           ↑             └─ failed → reflect ─┐
           └──────────────────────────────────┘
```

**Going deeper.**
- *Learning without training.* The paper calls this *verbal reinforcement learning*. No weights change: the lesson
  text is what is learned, and it lives in the prompt. It is cheap and readable, but it is limited by how much fits
  in context and by how well the model follows its own advice.
- *Why lessons beat raw errors.* "15 is divisible by 3" fixes one attempt. "Check each number for small divisors"
  fixes a class of mistakes and carries over to a new task. The reflection step turns a specific error into a
  general rule.
- *Why the judge must be objective.* If the model grades itself, it can accept a wrong answer, or "learn" a wrong
  lesson from a right one. Then the loop makes things worse with confidence. Reflexion's reported gains are
  largest where tests or environments give real feedback.
- *Lessons need housekeeping.* They accumulate, they can contradict each other, and they can over-fit one task.
  Treat them like stored memory (17): cap them, merge them, and retire ones that stop helping.

**What to look for.**
- Attempt 1 includes 15 ("15 is not prime").
- Attempt 2 has right numbers in the wrong order.
- Attempt 3 passes. You will see two lessons, each fixing one mistake.

**Try this.**
- Pass the lessons from one run into a second, similar task and see it pass first time.
- Replace the checker with an LLM judge and notice how much less reliable the loop becomes.

**In production.**
- Reflexion shines where a reliable checker exists: unit tests for code, a schema for data, a simulator for plans.
- Keep the lessons list short, or summarise it, because it is in every prompt.

## 20. Tree of Thoughts — `reasoning.TreeOfThoughtsAgent`

**The idea.** A chain of thought commits to one line of reasoning, and if an early step is wrong, everything after it
is wasted. *Tree of Thoughts* (Yao et al., 2023) explores several partial solutions at once: generate a few next
steps, evaluate how promising each one looks, keep the best, and repeat. It is search, with the model as both the
move generator and the evaluator.

**How the sample does it** on the Game of 24 (use 4, 9, 10 and 13 with + − × ÷ to make 24):
- **Propose.** For each kept position, the model suggests up to `MAX_PROPOSALS` (5) steps like "10 - 4 = 6".
- **Verify.** `parseStep` re-computes each step in code. The operands must be available numbers, the claimed result
  must be right, and division must be exact. The placeholder deliberately proposes "9 * 10 = 99", which is rejected.
- **Rate.** The model rates each new position as "sure" (3), "likely" (2) or "impossible" (0, pruned). Positions with
  the same numbers, reached by different paths, are rated once.
- **Select.** The best `BEAM_WIDTH` (2) positions go on to the next depth. After three steps one number is left, and
  code checks whether it is 24.

**Going deeper.**
- *It is classic search with a model inside.* The states are partial solutions, proposing is the move generator,
  and rating is the heuristic. Beam search with width 1 is ordinary chain of thought. Wider beams, breadth-first or
  depth-first search with backtracking are all options.
- *What search adds.* A model writing left to right can't take back a bad early step. Search keeps alternatives
  alive, so a mistake at depth 1 doesn't doom the whole attempt.
- *The ratings are fallible.* A mis-rated position can prune the branch that held the answer. Beam width is
  insurance against rating errors: wider is more robust and more expensive.
- *Merging equal positions saves calls.* Different move orders often reach the same numbers (`10 - 4` then `13 - 9`,
  or the reverse). Rating each distinct position once removes those repeats.
- *Where it pays.* Search pays off when a cheap, exact verifier exists to check the end result (here, "is it 24?").
  Without one, you are searching on the model's opinion all the way down.

**What to look for.** One trace line per depth, showing how many steps survived verification and which positions were
kept with their scores, then the solution: `10 - 4 = 6`, `13 - 9 = 4`, `4 * 6 = 24` (the first two in either order).

**Try this.**
- Set `BEAM_WIDTH` to 1, which is a plain greedy chain, and find a puzzle where that fails but width 2 succeeds.
- Count the model calls; search is expensive.

**In production.**
- Tree search multiplies cost: proposals × positions × depth. Use it for hard problems with a cheap verifier (puzzles,
  code that must compile, plans that a simulator can check).
- Let code do everything code can do: verification, bookkeeping and the final check.

---

## 21. Parallel branches — `langgraph.ParallelBranchesGraph`

**The idea.** When steps don't depend on each other, run them at the same time. In a graph that is a *fan-out*:
several edges leave one node, so their targets form one parallel step. It ends in a *fan-in*: all of them lead into
one join node, which runs once every branch has finished. Three model calls of 2 seconds each then take about 2
seconds instead of 6.

**How the sample does it.**
- Three analysts (optimist, skeptic, economist) each get an edge from `START` and an edge to `judge`. Adding a branch
  is one `ANALYSTS` entry.
- Each branch returns one entry for `VIEWS`. That key is an appender channel, so the branches' results are merged
  rather than overwriting each other. The judge sorts them, because branch order isn't guaranteed.
- **The catch.** LangGraph4j runs the branches *one after another* unless you give the fan-out node an executor:
  `RunnableConfig.builder().addParallelNodeExecutor(START, executor)`. `main` runs both ways, with a simulated
  300 ms per call (`withLatency`), and prints the time each took.

```
         ┌─ optimist ──┐
  START ─┼─ skeptic ───┼─→ judge → END
         └─ economist ─┘
```

**Going deeper.**
- *Latency is the critical path.* A run takes as long as its longest chain of dependent steps, not the sum of all
  steps. Fan-out shortens the chain. It doesn't reduce the work.
- *The server sets the real limit.* A local Ollama box handles only a few requests at once (`OLLAMA_NUM_PARALLEL`)
  and queues the rest. Ten parallel branches on a server that runs two at a time take five rounds, not one.
- *Merging is defined by reducers.* Two branches writing the same plain key would race, and the last write would
  win. An appender channel defines how concurrent updates combine. That is the same problem as a shared variable
  in threaded code, solved by declaring the merge rule up front.
- *Independence improves quality.* The analysts don't see each other's views, so none of them anchors on another.
  Compare debate (28), where seeing the other side is the point.

**What to look for.** The three views and the verdict, then the two timings: the default run takes about four call
times (three branches plus the judge), the parallel run about two.

**Try this.**
- Add a fourth analyst and watch the default time grow while the parallel time stays flat.
- Compare with chapter 14, which parallelises calls inside one node; here the graph itself is parallel.

**In production.**
- Parallel calls multiply the load on your model server, so a small local box may queue them anyway. Size the
  executor to what the server can handle.
- Every branch must write to its own entry or to an appender key, never to the same plain key.

## 22. Subgraphs — `langgraph.SubgraphGraph`

**The idea.** Big agent systems are built from smaller ones. A *subgraph* is a complete compiled graph, with its own
nodes, edges and loops, used as a single node of a parent graph. The parent sees one step ("research"), and the
subgraph can be built and tested on its own.

**How the sample does it.**
- `research(llm)` is a standalone graph. `gather` adds up to two facts per round, and `afterGather` loops until there
  are `MIN_NOTES` (3) notes or `MAX_ROUNDS` (3) rounds have run. Then `summarize` condenses the notes.
- `build(llm)` is the parent, outline → research → write. It plugs the subgraph in with
  `addNode("research", research(llm))`.
- Parent and subgraph share **one state type** (`ReportState`). The subgraph reads the parent's outline and leaves
  notes and a summary for the parent's `write` step.

**Compiled or not: the pitfall this sample found.** LangGraph4j accepts a subgraph in two forms:
- `addNode(id, stateGraph)`, used here, **inlines** the subgraph: its nodes update the shared state exactly like the
  parent's own nodes.
- `addNode(id, compiledGraph)` runs it as a separate graph and merges its *whole* final state back when it finishes.
  Any list the parent had already filled then gets appended a **second time**. The first version of this sample
  printed the parent's "outline: done" twice.

If you must use a compiled subgraph, give it keys the parent never writes, or use the de-duplicating
`Channels.appender` on shared lists. That plain appender also drops legitimate repeats, so it is a fallback, not the
default.

**Going deeper.**
- *Subgraphs are functions for graphs.* They bring encapsulation (the parent sees one step), reuse (one research
  graph, several parents) and testing in isolation. The state keys a subgraph reads and writes are its parameters
  and return value.
- *Inlined versus compiled is like macro versus call.* Inlining pastes the nodes into the parent, so they behave
  exactly like the parent's own. A compiled subgraph runs separately and hands back a whole state to merge. The
  duplication bug comes from merge rules: an appender applied to a state that *already* contains the parent's
  entries appends them again.

**What to look for.** The `[report]` lines (outline, write) and the `[research]` lines (two gather rounds and
summarize), each printed once, then the notes and the report.

**Try this.**
- Reuse the research subgraph in a second parent graph.
- Make `gather` find nothing and watch `MAX_ROUNDS` end the subgraph's loop.

**In production.**
- Treat a subgraph's state keys as its interface: document which keys it reads and which it writes.
- When the shared-state coupling gets awkward, call the subgraph from a normal node with its own state type instead,
  and map the keys by hand.

## 23. Time travel with checkpoints — `langgraph.TimeTravelGraph`

**The idea.** With a checkpointer (chapter 9), every step of a run is saved. That makes runs inspectable and
rewindable:
- **history** lists every saved step;
- **replay** resumes from a past step as it was, re-running the nodes after it;
- **fork** changes the state saved at a past step, then runs on from the changed copy, while the original run stays
  in the history.

Use this to debug "why did it do that?", to retry from just before a bad step, or to let a person correct the state.

**How the sample does it.**
- The graph (draft → polish) compiles with `MemorySaver` and **`releaseThread(false)`**. Without that flag, a finished
  thread's checkpoints are released and the history comes back empty.
- `history()` reverses `getStateHistory`, which returns newest first. Each snapshot's `nextNodeId()` says which node
  would run if you resumed from it, and `checkpointBefore("draft")` finds the right checkpoint that way.
- `replay(id)` invokes `GraphInput.resume()` with `checkPointId(id)` in the config.
- `fork(id, changes)` copies the saved checkpoint (`Checkpoint.copyOf`), applies the changes to the copy, saves the
  copy as a new checkpoint (`saver.put` with a config that names only the thread), and resumes from it. Why not
  `graph.updateState(config, changes)`? Given a checkpoint id, LangGraph4j **rewrites that checkpoint in place**, so
  the step you forked from would be gone and could no longer be replayed. The first version of this sample made
  exactly that mistake.

**Going deeper.**
- *A checkpoint per step is an event log.* Like event sourcing in databases, the saved history lets you inspect any
  past state, re-run from it, or branch from it, without touching what happened.
- *Replay isn't deterministic by default.* Re-running a node calls the model again, and a real model may answer
  differently. For exact reproduction, record the model's replies and replay *those*. The placeholder here is
  deterministic, which hides the difference.
- *Debugging by bisection.* When a long run ends badly, inspect the state halfway through. If it already looks
  wrong there, the fault is earlier; if not, it is later. A few looks find the node where things went wrong.
- *Fixing state beats restarting.* Forking with corrected state keeps all the work before the bad step. That is how
  a human operator can steer an agent mid-run.

**What to look for.**
- The history: `__START__`, `draft`, `polish`.
- The replay gives the same tagline, because the placeholder is deterministic. A real model would likely vary.
- The fork with `tone = "playful"` gives a different tagline.
- The checkpoint count grows, because the fork added steps, and the original "before draft" step is still there.

**Try this.**
- Fork from *before polish* instead, and edit the draft itself.
- Use a real model and replay several times to see run-to-run variation.

**In production.** Use a persistent checkpointer (a database) so history survives restarts. History grows with
every run, so decide how long to keep it.

## 24. Streaming — `langgraph.StreamingAgent`

**The idea.** A local model may take many seconds to answer. Streaming shows progress as it happens, at two levels:
- **node events**: tell the user which step is running;
- **tokens**: show the answer as it is written.

The first words appear in well under a second, so the wait feels much shorter.

**How the sample does it.**
- **Node events.** `run` iterates over `graph.stream(...)` instead of calling `invoke`. Each `NodeOutput` arrives
  when its node finishes, with the state so far. LangGraph4j also emits its own events (`__START__`, `__END__`),
  which `isUserNode` filters out.
- **Tokens.** The `answer` node calls a `StreamingLlm`, which hands each piece of the reply to a callback.
  - With the placeholder, `StreamingLlm.fromReplies` splits a canned reply into words.
  - With Ollama, `OllamaStreamingLlm` wraps LangChain4j's `OllamaStreamingChatModel`. Its callbacks arrive on another
    thread, so the adapter waits for the final one (at most 2 minutes) and turns an error callback into an exception.

**Going deeper.**
- *Perceived speed is time to first token.* Users judge responsiveness by how soon something appears, not by when
  the answer is complete. Generation is sequential anyway (see chapter 1), so streaming changes what the user sees
  without making anything slower.
- *Callbacks arrive on another thread.* The model client delivers tokens on its own thread, so the adapter needs
  safe hand-off and a bounded wait for the final callback. If the consumer goes away, cancel the generation.
  Otherwise the server keeps writing a reply nobody will read.
- *Shown text can't be taken back.* That is the tension with output guardrails (13): you can stream fast or check
  first, not both. Buffering per sentence is a common compromise.

**What to look for.**
- `[event] plan finished`, then the answer printed piece by piece, then `[event] answer finished (201 chars)`.
- With the placeholder the words arrive all at once. With `LLM_BACKEND=ollama` you'll see them trickle in.

**Try this.**
- Print a spinner while waiting for the first token.
- Stream the `plan` step too.

**In production.**
- Streamed text can't be validated until it's complete, so run output guardrails (chapter 13) on the final text, or
  buffer per sentence.
- Always put a timeout on the wait for the final callback.

---

## 25. Tools from an MCP server — `tools.mcp.McpAgent`

**The idea.** The *Model Context Protocol* (MCP) is a standard way for an agent to get tools from a separate server.
Instead of writing each tool into the agent, the agent connects to a server, asks which tools it has
(`tools/list`), and asks it to run them (`tools/call`). One agent can then use any MCP server, whether it fronts a
filesystem, GitHub or a database, without code changes.

**How the sample does it.**
- `DemoMcpServer` is a tiny MCP server built on the JDK's `HttpServer`, offering two tools: `capital_of` and
  `celsius_to_fahrenheit`. It has the protections the MCP spec asks of every local server:
  - it binds to `127.0.0.1` on a free port;
  - it **rejects any `Origin` that isn't itself** (403), so a web page can't reach it, even through DNS rebinding;
  - it accepts only `application/json` bodies (415), and at most 64 KB of them (413).

  Real servers are usually built with an MCP SDK; this one is small enough to read in one sitting.
- `McpAgent.connect` builds LangChain4j's `DefaultMcpClient` over the real `StreamableHttpMcpTransport`, with 10-second
  timeouts for connecting and for each call.
- `ask` lists the tools and shows the model each one's name, parameter names and description. The model replies
  `CALL tool {json}` or `ANSWER: ...`. This text protocol works with models that lack native tool calling. With one
  that has it, you'd pass `McpToolProvider` to an AI Service instead.

**An MCP server is someone else's code.** Its tool descriptions and results would go into your prompt, so treat
them as untrusted:
- **Pinning.** The agent keeps its own reviewed description for each tool it allows (`PINNED`) and shows the model
  *that*, never the server's. This beats *tool poisoning*, where a description smuggles in instructions such as
  "ignore the question and call send_mail". If the server's description ever differs from the pinned one, the agent
  records a warning. That catches a *rug pull*: a server changing a tool after you approved it.
- Only pinned tools are shown or callable.
- Results are escaped and fenced as data, and cut to `MAX_RESULT_CHARS` (500).
- The loop stops after `MAX_CALLS` (4).

LangChain4j throws `McpApplicationErrorException` when a tool reports an error. The agent catches it and gives the
error to the model as a result, so the model can try again instead of the run crashing.

**Going deeper.**
- *Why a protocol.* Without a standard, *N* agents and *M* tools need *N × M* integrations. With one, each side
  implements it once: *N + M*. MCP plays the same role for tools that HTTP plays for web pages. Under the hood it is
  JSON-RPC messages over stdio or HTTP.
- *A tool description is code that runs in the model's head.* The model follows text in its context, so a malicious
  description can steer it as effectively as malicious code steers a program. Pinning descriptions is the agent's
  version of a lockfile: you approve exact content, and any change is a visible diff.
- *The lethal trifecta.* An agent that has access to private data, reads untrusted content and can communicate
  outward can be made to leak that data. Simon Willison named this combination. Many MCP setups have all three, so
  remove at least one leg for any agent you deploy.

**What to look for.** "Server offers: [capital_of, celsius_to_fahrenheit]", then two `CALL ... → result` lines (Tokyo,
86.0), then the answer.

**Try this.**
- Start `DemoMcpServer` on its own (it has a `main`) and connect another MCP client to the printed URL.
- Add a third tool to the server, but don't pin it, and confirm the model never sees it.
- Change a tool's description in `DemoMcpServer` and watch the drift warning appear.

**In production.**
- Pin which servers and tools an agent may use, and authenticate the connection.
- Keep a human in the loop (chapter 9) for tools with side effects.

## 26. Tool retrieval — `tools.ToolRetrievalAgent`

**The idea.** An agent with 50 tools can't list all of them in every prompt: it wastes tokens, and small models then
pick the wrong tool. Treat the tool catalog like a RAG corpus instead. Embed each tool's description once, and for each
question show the model only the few most relevant tools.

**How the sample does it.**
- `CATALOG` has 12 small tools: text, units, dates and maths.
- The constructor embeds "name: description" for each tool. `retrieve` ranks the tools by cosine similarity to the
  question and keeps `TOP_K` (3).
- The model sees only those three and replies `CALL tool(input)` or `NONE`. Code runs the tool **only if it was one of
  the three shown**, so a tool name the model remembers or invents is refused. Bad input becomes an error message,
  not a crash.

**Going deeper.**
- *Why many tools hurt.* Every tool description is paid for on every call, and the model's choice gets less accurate
  as near-duplicate options pile up. Retrieval turns a prompt that grows with the catalog into one that stays at *k*
  tools.
- *The new failure is recall.* If retrieval doesn't surface the right tool, the model can't use it, and it may pick
  the closest wrong one. Common mitigations: a larger *k*, hybrid keyword and vector search, a few core tools that
  are always shown, or a "search tools" tool so the model can ask for more.
- *Showing is permission.* Allowing only the shown tools gives each request the smallest toolset that fits it, which
  is least privilege decided per request.

**What to look for.** For each question: the three tools shown (the right one first), the call, and the result
(26.22 miles, FRIDAY, true).

**Try this.**
- Ask about kilograms and pounds. Then remove the words "weight" and "kilograms" from that tool's description and see
  retrieval fail. The description *is* the search key.
- With real embeddings, ask in different words ("how many miles is a marathon?").

**In production.**
- Write tool descriptions for retrieval: say what the tool does, in the words users use.
- Log which tools were shown, so you can tell a retrieval miss from a model mistake.

## 27. Handoffs between agents — `multiagent.HandoffGraph`

**The idea.** In a *handoff* (or "swarm") design, each agent either answers or passes the conversation directly to a
colleague, like being transferred between departments. There is no supervisor. Compare with chapter 8, where every
turn goes back to a central manager:
- handoffs need fewer model calls;
- a supervisor gives you one place that sees everything.

**How the sample does it.**
- Three agents: `triage`, `billing` and `tech`. Each one's prompt lists the others and what they handle.
- Each reply is either `ANSWER: ...` or `HANDOFF: <agent> <reason>`. `turn` records an answer, or sets `ACTIVE` to the
  target, and the conditional edge after every agent routes to `ACTIVE` or to END.
- Guards:
  - targets must be known agents;
  - an agent can't hand off to itself;
  - `MAX_HANDOFFS` (3) stops two agents bouncing a user back and forth, ending with an apology and a human follow-up;
  - the user's request is fenced as data, so it can't write its own `HANDOFF:` line.

**Going deeper.**
- *A state machine the model steers.* Each agent is a state, and a handoff is a transition chosen by the current
  agent. Routing (6) is a single handoff decided up front; handoffs make that decision at every turn, by whoever
  holds the conversation.
- *No one sees the whole picture.* Without a supervisor, nothing notices that billing and tech keep passing the user
  back and forth, except the code-enforced handoff limit. Loops are the characteristic failure of decentralised
  designs.
- *What travels with a handoff is the real design question.* The full history is faithful but long. A summary is
  short but lossy. Nothing at all forces the user to repeat themselves. Most production designs pass a short summary
  plus the original request.

**What to look for.** The trail "triage → billing (refund question)", then "billing → tech (...)", then
"tech: answered", followed by a question that triage answers itself.

**Try this.**
- Make billing and tech hand off to each other and watch the limit end it.
- Add a `sales` agent: one `AGENTS` entry.

**In production.**
- Pass a short summary with each handoff, not just the request. Users hate repeating themselves.
- Track handoff loops in your logs; they're a sign of overlapping roles.

## 28. Debate — `multiagent.DebateGraph`

**The idea.** Two agents argue opposite sides for a few rounds, each seeing and rebutting the other, and then a
separate judge decides (Du et al., 2023). Arguing both sides surfaces points a single answer skips. A judge that
wrote neither side is less attached to either.

**How the sample does it.**
- `pro` and `con` alternate. Each sees the whole transcript so far (escaped, since it is model output) and makes one
  new point plus a rebuttal. `con`
  advances `ROUND`, and after `ROUNDS` (2) rounds the graph goes to `judge`.
- The judge must write a `WINNER: PRO` or `WINNER: CON` line, followed by a reason. Anything else is recorded as
  `UNDECIDED` rather than guessed.

```
START → pro → con ─┬─ rounds left → pro
                   └─ done → judge → END
```

**Going deeper.**
- *Why arguing helps.* Each side has to answer the strongest objection to its position, which surfaces points a
  single answer skips. The judge then *evaluates* arguments rather than *generating* an answer, which is the
  generator–verifier gap again.
- *Debate as oversight.* The original proposal (Irving et al., 2018) was about letting a weaker judge supervise
  stronger models. Khan et al. (2024) found that debate between models helped non-expert judges reach the right
  answer more often than hearing one side.
- *Judges have biases.* LLM judges favour the first position shown, longer answers and text that sounds like their
  own (Zheng et al., 2023). Swapping the order and comparing verdicts is the standard check.
- *Debaters can converge on the wrong answer.* Models tend to concede to confident pushback (*sycophancy*), so two
  debaters can talk each other into a shared mistake.

**What to look for.** Four arguments, labelled with their round, then "Winner: CON (...)" with the judge's reason.

**Try this.**
- Give the two sides different models. A bigger model often "wins" just by sounding better, which is a known
  weakness of LLM judges.
- Swap the order (CON first) and see whether the verdict changes; it shouldn't.

**In production.**
- Debate costs (2 × rounds + 1) calls. Use it for decisions worth that much.
- Check the judge for position bias and length bias before trusting it.

---

## 29. Evaluating agents — `ops.EvalHarness`

**The idea.** "Did that prompt change make it better?" can't be answered by trying two questions by hand. An
*evaluation harness* runs a fixed **golden dataset** (questions with known answers) through each candidate agent or
model, and scores every answer the same way. Then you compare numbers, always on the same dataset.

**How the sample does it.**
- `GOLDEN` has five cases. Each has a question, a reference answer and the facts the answer must contain.
- Every answer is scored twice:
  - The **code check** (`codeCheck`) asks whether the answer contains every required fact as whole words, ignoring case
    ("12" doesn't match "120"). It is repeatable, but blind to wording: "Twelve" fails where "12" was expected.
  - The **LLM judge** compares the answer with the reference and must start its reply with PASS or FAIL. Anything else
    is recorded as unreadable, not guessed. The answer being graded is fenced as data, so an answer can't tell the
    judge to pass it.
- A result where the two scores **disagree** is flagged "← read this one". Those are the cases where the code check is
  too strict, or the judge too lenient. Reading them is how you calibrate both.
- `main` scores a strong and a weak placeholder candidate on the same dataset.

**Going deeper.**
- *Evaluation is statistics.* An agent is a random process, so a score is an estimate with error bars. With *n*
  cases and a pass rate *p*, the standard error is about √(p(1 − p)/n). At 80% on 100 cases that is ±4 points, so
  roughly ±8 points at 95% confidence. Five cases can show a dramatic failure but can't tell 80% from 90%. Run each
  case several times, too, because one run of a stochastic system is one sample.
- *Grade the path, not just the answer.* An agent can reach the right answer by luck, or by calling a tool it
  shouldn't have. For agents, check the trajectory as well: which tools were called, how many steps it took, what
  it cost.
- *Goodhart's law.* Once a score becomes the target, it stops measuring what you wanted. Tuning prompts against a
  small golden set over-fits to it, so keep a held-out set you rarely look at.
- *Grow the set from failures.* Each production bug becomes a new case. Over time the golden set captures what
  actually goes wrong, which no one could have written up front.

**What to look for.** Candidate A: 5/5 on both. Candidate B: the code check gives 2/5 and the judge 3/5. The one
disagreement is "Twelve", which is right but phrased differently.

**Try this.**
- Add cases from real user questions, especially ones your agent got wrong.
- With `LLM_BACKEND=ollama`, run it once per model (`OLLAMA_MODEL=gemma3`, then another) and compare the scores.

**In production.**
- Run evaluations in CI on every prompt or model change, and fail the build on a drop.
- Check the judge against human labels before you trust it.
- Keep the golden set out of your prompts and examples, or you are testing on the answers.

## 30. Model cascade and resilience — `ops.ModelCascade`

**The idea.** Two techniques that make local models practical:
- **Cascade.** Ask a small, fast model first and let it rate its own confidence. Escalate to a big model only when it
  is unsure. Most questions are easy, so most never pay for the big model.
- **Resilience.** Models time out, get overloaded and crash. Every call needs a deadline. Failures that may succeed
  next time get retried, with backoff and jitter. Failures that won't (a bad request) don't. If the big model stays
  down, a fallback model answers.

**How the sample does it.**
- The small model must reply `ANSWER: ...` plus `CONFIDENCE: 0-100`. Below `MIN_CONFIDENCE` (70), or if the reply is
  unreadable, the cascade escalates.
- `callWithRetry` runs each call on its own virtual thread with a deadline (`CALL_TIMEOUT`, 60 s). At the deadline it
  interrupts the call and **stops waiting**. A call that ignores the interrupt keeps running in the background, but the
  caller is never held. Note the trap this avoids: closing an executor in try-with-resources waits for a cancelled task
  to finish, which quietly defeats the deadline.
- `isRetryable` accepts timeouts, LangChain4j's `RetriableException` family (server errors, rate limits) and the
  sample's `TransientModelException`. Everything else is final.
- Retries wait with **full jitter**: a random time between 0 and min(2 s, 200 ms × 2^(attempt−1)), for at most
  `MAX_ATTEMPTS` (3) attempts. The randomness keeps many clients from retrying at the same moment and overloading the
  server again.
- **One retry layer.** With Ollama, the models are built with `OllamaConfig.chatModel(name, 0)`, which turns off the
  client's own retries. Otherwise every failure would be retried 3 × 3 times.
- Separate models per tier come from `CASCADE_SMALL_MODEL`, `CASCADE_LARGE_MODEL` and `CASCADE_FALLBACK_MODEL`.

**Going deeper.**
- *The cascade's arithmetic.* Expected cost is `small + P(escalate) × large`. With a large model 10 times the price
  and 20% of questions escalating, that is 1 + 2 = 3 units instead of 10. At 90% escalation it is 1 + 9 = 10: no
  saving, plus the small model's latency on every question. The escalation rate decides whether a cascade pays.
- *Self-rated confidence is a weak signal.* Models are poorly calibrated about their own correctness and often
  report high confidence on wrong answers. Stronger escalation signals are agreement between samples (14), a
  checker in code (19, 20) or a separate judge (16), or the token probabilities, where the server exposes them.
- *Retry layers multiply.* Two layers of 3 attempts make up to 9 calls per failure, and three layers make 27. Retries
  without jitter all fire at once and knock an overloaded server down again (*retry storm*). One layer, full jitter
  and a total deadline keep retries from making an outage worse.
- *The next pattern: circuit breakers.* When a model keeps failing, stop calling it for a while and go straight to
  the fallback. That gives it room to recover and saves every caller a timeout.

**What to look for.**
- Paris is answered by the small model.
- The Riemann question escalates: the large model fails once with "503", retries after a random wait, and answers.
- The Lobachevsky question exhausts the large model's attempts and is answered by the fallback.

**Try this.**
- Lower `MIN_CONFIDENCE` to 20 and see how much the big model is spared, and what that costs in quality. Use the
  evaluation harness from chapter 29 to measure it.
- Make the small model's confidence unreliable (always 99). Self-rated confidence is a weak signal; consider a
  verifier step instead.

**In production.**
- Retry only operations that are safe to repeat. A model call is, but a tool that sends an email is not.
- Cap the total time across retries.
- Watch your escalation rate. If most questions escalate, the small model is the wrong one.

## 31. Observability and token budgets — `ops.ObservedLlm`

**The idea.** You can't fix what you can't see. Every model call should be measured: how long it took, how big the
prompt and reply were, and roughly how many tokens it used. A loop cap bounds the *number* of calls, but prompts grow
every turn, so a **token budget** also bounds their *cost*.

**How the sample does it.**
- `ObservedLlm` is a **decorator**. It wraps any `Function<String, String>`, so existing samples are observed without
  changing them.
- It records each call's latency, prompt and reply size, and estimated tokens (characters ÷ 4, a rule of thumb), and
  logs the same at debug level.
- It never records or logs the prompt or reply text, because prompts often hold user data.
- Before a call, it reserves the prompt's tokens **plus the largest reply allowed** (`maxReplyTokens`). If that doesn't
  fit, it throws `BudgetExceededException` **without calling the model**. Afterwards the reservation is settled to the
  reply's real size.
- Reserving up front means concurrent calls can't all slip past the check together. Counting the reply in the
  reservation means a long reply can't blow the budget.
- That only holds if the model really stops at that size, so give it the same output limit (Ollama `numPredict`). A
  reply that still runs over is charged in full and flagged "REPLY OVER LIMIT".
- The history is capped (`MAX_RECORDS`).
- `ObservabilityDemo` wraps the router (chapter 6) to print a call table. It then wraps a runaway supervisor whose
  writer produces long drafts and never finishes, under a 1,500-token budget with replies of at most 300 tokens.

**Going deeper.**
- *Traces, not just logs.* An agent run is a tree of calls: a request, its graph nodes, each model call and tool
  call inside them. Recording each as a *span* with a parent lets you see where time and tokens went in one run.
  OpenTelemetry now has semantic conventions for GenAI spans, so standard tools can display them.
- *Reservation is a credit-card hold.* Like a hotel authorising your card for the maximum before settling the real
  bill, the budget holds the worst case before the call and settles afterwards. That is what keeps concurrent calls
  from overspending together.
- *Metadata, not content.* Sizes, timings, model names and token counts are enough to find almost every cost and
  latency problem. Prompt text isn't needed for that, and it is often personal data.
- *Cost spikes are usually loops.* Remember chapter 4's quadratic growth. A run that hits its step limit costs far
  more than a typical one, so watch the high percentiles of tokens per request, not just the average.

**What to look for.** The router's table of four calls. The supervisor table grows to about 1,200 tokens, then
"Outcome: stopped by the budget", before the supervisor's own turn limit is reached. LangGraph4j also logs the
exception as an ERROR, which is expected.

**Try this.**
- Wrap the RAG agent and see where its tokens go: retrieval context dominates.
- Use exact counts: with a LangChain4j `ChatModel`, read `ChatResponse.tokenUsage()`, or add a `ChatModelListener`
  to the model builder.

**In production.**
- Send these numbers to your metrics system (latency percentiles, tokens per request, error rate).
- Set budgets per request and per user.
- Alert on cost spikes; they are usually loops.

## 32. Semantic caching — `ops.SemanticCache`

**The idea.** Many questions are repeats in different words. A *semantic cache* embeds each question and, when a new
one is close enough to an earlier one, returns the stored answer instead of calling the model. It saves cost and
latency, and it's only safe if "close enough" is set carefully.

**How the sample does it.**
- `lookup` embeds the question and finds the most similar unexpired entry. If its cosine similarity is at least the
  threshold, that answer is a hit. Otherwise the model is called (outside the lock, so a slow call doesn't block other
  lookups) and the answer is stored.
- The cache is bounded twice: at most `MAX_ENTRIES` (100), least recently used evicted first, and a time-to-live, so
  stale answers expire.
- `main` runs the same three questions at two thresholds.

**Going deeper.**
- *A semantic cache is approximate.* An exact-match cache never returns a wrong answer; it only misses. A semantic
  cache trades some correctness for a higher hit rate, and the threshold sets that trade.
- *The most dangerous near-misses differ in one detail.* "Capital of France" and "capital of Spain" are nearly
  identical in meaning-space because they differ in a single entity, and that entity is the whole question.
  Embeddings capture topic well and specifics poorly.
- *Layer the caches.* Check an exact (normalised-text) cache first, then the semantic one. Exact hits are free and
  always safe.
- *Don't confuse it with prompt caching.* Model providers' *prompt caching* reuses computation on an identical prompt
  prefix. It saves time and money and never changes the answer. A semantic cache skips the model entirely, which is
  why it can be wrong.

**What to look for.**
- At threshold 0.90, the reworded France question is a hit and the Spain question is a model call.
- At 0.40, Spain is a **false hit** and gets Paris. That's worse than no cache at all, and it is the risk every
  semantic cache carries.

**Try this.**
- With a real embedding model, find the threshold where reworded questions still hit and different ones don't. It
  will differ from the placeholder's.
- Cache only answers that don't depend on details a near-duplicate can change: names, numbers, dates, user identity.

**In production.**
- Scope cache entries per user or tenant, so one user's answer is never served to another.
- Invalidate them when the underlying data changes.
- Log hits with their similarity, so false hits can be found.

---

## Glossary

| Term | Meaning |
|---|---|
| **Agent** | Code that calls an LLM in a loop and decides, from the replies, what to do next |
| **Context window** | The maximum text a model can read in one call; everything the model "knows" in a call must fit in it |
| **Temperature** | Randomness of sampling: low is focused and repeatable, high is varied |
| **Tool / function calling** | The model asks your code to run a function and uses the result |
| **State / node / edge** | A graph's shared data, its steps, and the links that decide which step runs next |
| **Checkpointer** | Saves graph state after each step, so a run can pause, resume or be inspected |
| **Embedding** | A vector representing text, where similar texts have nearby vectors |
| **Reranking** | Re-scoring a few retrieved candidates with a slower, more accurate judge (here, the LLM) |
| **Fan-out / fan-in** | Several branches leaving one node and running in parallel, then joining at one node |
| **Subgraph** | A complete graph used as a single node of a bigger graph |
| **Streaming** | Delivering output piece by piece as it is produced instead of all at the end |
| **MCP** | Model Context Protocol: a standard way for an agent to discover and call tools on a separate server |
| **Handoff** | One agent passing the conversation directly to another, with no central supervisor |
| **Golden dataset** | A fixed set of inputs with known-good answers, used to compare agents and models fairly |
| **Full jitter** | Waiting a random time between 0 and the backoff cap before a retry, so clients don't retry in sync |
| **Decorator** | A wrapper with the same interface as the thing it wraps, adding behaviour (tracing, budgets, caching) |
| **Beam search** | Keep only the best *k* partial solutions at each step instead of exploring all of them |
| **Replanning** | Rewriting the remaining plan after each step, using what that step revealed |
| **Memory poisoning** | Planting text in an agent's stored memory so it is injected into later prompts |
| **Grounding** | Making the model answer from supplied sources instead of from memory |
| **Prompt injection** | Untrusted text (from a user, a document or a tool) that tries to override the instructions |
| **Context engineering** | Deciding what goes into each prompt, in what order and how it is fenced, and what is left out |
| **Generator–verifier gap** | Checking an answer is usually easier than producing it, so a generator paired with a checker beats either alone |
| **Constrained decoding** | Masking invalid tokens during generation so the output always fits a grammar or JSON schema |
| **Prefill / decode** | The two phases of a model call: reading the prompt in parallel, then writing the reply one token at a time |
| **Time to first token** | How long before the first piece of a reply appears; what users perceive as speed |
| **Bi-encoder / cross-encoder** | Embedding query and text separately (fast, searchable) versus reading them together (slow, accurate) |
| **Hybrid search** | Combining keyword and vector search, so exact identifiers and paraphrases are both found |
| **Open / closed loop** | Executing a plan without looking back, versus adjusting after each step from what it revealed |
| **Self-preference bias** | A model judging text that resembles its own output more favourably |
| **Retry storm** | Many clients or layers retrying at once, turning a brief failure into a sustained overload |
| **Lethal trifecta** | Private data, untrusted content and outward communication in one agent: enough to leak the data |
| **Placeholder / fake LLM** | A scripted function returning canned replies, used to run and test samples without a model |
