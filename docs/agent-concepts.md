# Agent concepts, one sample at a time

This guide walks through every sample in the repository. Each chapter explains one idea behind LLM agents, shows
how the sample implements it, tells you what to look for when you run it, and suggests changes to try. It ends with
what tends to go wrong in real systems. Read it next to the code: every chapter names the class to open.

- [How the samples are built](#how-the-samples-are-built)
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

## Suggested reading order

Start with chapters 1–4 to see a model call, tools, a loop and the ReAct pattern. Then read 5–9 in order: each
workflow pattern adds one idea to the previous one. Chapters 10–14 are independent; read them when you need them. Read 15–17 after chapter 12, and 18–20 after chapter 7.

---

## 1. Calling a model — `chat.HelloOllama`

**The idea.** The smallest possible program: build a chat model client, send one prompt, and print the reply.
Everything else in this repository is code around this one call.

**How the sample does it.** `OllamaConfig.chatModel()` builds a LangChain4j `OllamaChatModel` from environment
variables, with a low temperature (0.2) and a 2-minute timeout. `model.chat(prompt)` returns the reply text.

**What to look for.** The model name and the reply. Run it twice: with temperature 0.2 the answers are similar but
rarely identical.

**Try this.**
- Change the temperature in `OllamaConfig` to 1.0 and compare a few runs.
- Ask the same question to two different models (`OLLAMA_MODEL=...`).

**In production.** Every outbound call needs a timeout; this one has 2 minutes. Never hard-code the URL or the
model name; here they come from the environment.

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
- The top `TOP_K` (2) documents go into `<doc id="...">` tags, with instructions to cite `[doc-id]` and a fixed
  refusal sentence.
- If retrieval finds nothing, the agent answers "I don't know" **without calling the model**, the cheapest defence
  against a confident made-up answer.

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
- Chunks go into the answer prompt escaped (`PromptText.escape`) and labelled as data, so a document containing
  `</doc>` can't break out of its fence.

**Watch the score scale.** LangChain4j store scores are *relevance* scores, `(cosine + 1) / 2`. A score of 0.5 means
unrelated, not "half relevant", so useful thresholds sit above 0.5.

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
  explicit "yes", so an unclear reply means "not relevant".
- `afterGrading` routes to `generate` if any chunk passed, to `rewrite` while fewer than `MAX_REWRITES` (2) rewrites
  have been made, and to `give_up` otherwise.
- `rewrite` asks for plainer or more formal words, falling back to the question if the reply is blank.

```
START → retrieve → grade ─┬─ some relevant ────────→ generate → END
           ↑              ├─ none, rewrites left ─→ rewrite ─┐
           └──────────────┼──────────────────────────────────┘
                          └─ none, no rewrites left → give_up → END
```

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
| **Beam search** | Keep only the best *k* partial solutions at each step instead of exploring all of them |
| **Replanning** | Rewriting the remaining plan after each step, using what that step revealed |
| **Memory poisoning** | Planting text in an agent's stored memory so it is injected into later prompts |
| **Grounding** | Making the model answer from supplied sources instead of from memory |
| **Prompt injection** | Untrusted text (from a user, a document or a tool) that tries to override the instructions |
| **Placeholder / fake LLM** | A scripted function returning canned replies, used to run and test samples without a model |
