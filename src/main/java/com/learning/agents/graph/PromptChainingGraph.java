package com.learning.agents.graph;

import com.learning.agents.llm.LlmProvider;
import com.learning.agents.llm.ScriptedLlm;
import org.bsc.langgraph4j.CompiledGraph;
import org.bsc.langgraph4j.GraphStateException;
import org.bsc.langgraph4j.StateGraph;

import java.util.Map;
import java.util.function.Function;

import static org.bsc.langgraph4j.StateGraph.END;
import static org.bsc.langgraph4j.StateGraph.START;
import static org.bsc.langgraph4j.action.AsyncEdgeAction.edge_async;
import static org.bsc.langgraph4j.action.AsyncNodeAction.node_async;

/**
 * Prompt chaining: split one task into a fixed sequence of small LLM calls, where each call works on the output
 * of the previous one. A plain-code <em>gate</em> between steps checks the intermediate result, so a bad outline
 * is retried (up to {@link #MAX_OUTLINE_ATTEMPTS}) instead of being passed down the chain.
 *
 * <pre>
 *   START → outline → gate ─┬─ valid ───────→ draft → polish → END
 *              ↑            ├─ invalid, retry ─┘ (back to outline)
 *              └────────────┘
 *                           └─ invalid, limit reached → END
 * </pre>
 */
public class PromptChainingGraph {

    static final int MAX_OUTLINE_ATTEMPTS = 2;
    static final int MIN_OUTLINE_POINTS = 3;

    /**
     * Builds the chain around a pluggable LLM so it can be tested without Ollama.
     *
     * @param llm maps a prompt to a model reply
     */
    public static CompiledGraph<PromptChainState> build(Function<String, String> llm) throws GraphStateException {
        return new StateGraph<>(PromptChainState.SCHEMA, PromptChainState::new)
                .addNode("outline", node_async(state -> {
                    String outline = llm.apply("""
                            Create an outline for a short blog post about "%s".
                            Reply with %d to 5 bullet points, one per line, each starting with "- "."""
                            .formatted(state.topic(), MIN_OUTLINE_POINTS));
                    int attempts = state.outlineAttempts() + 1;
                    return Map.of(
                            PromptChainState.OUTLINE, outline,
                            PromptChainState.OUTLINE_ATTEMPTS, attempts,
                            PromptChainState.LOG, "outline: attempt %d, %d point(s)".formatted(attempts, countPoints(outline)));
                }))
                .addNode("draft", node_async(state -> {
                    String draft = llm.apply("""
                            Write a draft of about 120 words that covers every point of this outline, in order.
                            Outline:
                            %s""".formatted(state.outline()));
                    return Map.of(PromptChainState.DRAFT, draft, PromptChainState.LOG, "draft: written");
                }))
                .addNode("polish", node_async(state -> {
                    String polished = llm.apply("""
                            Polish this draft: fix grammar, tighten wording, and add a one-line title. Keep the meaning.
                            Draft:
                            %s""".formatted(state.draft().orElse("")));
                    return Map.of(PromptChainState.FINAL_TEXT, polished, PromptChainState.LOG, "polish: done");
                }))
                .addEdge(START, "outline")
                .addConditionalEdges("outline",
                        edge_async(PromptChainingGraph::gate),
                        Map.of("valid", "draft", "retry", "outline", "give_up", END))
                .addEdge("draft", "polish")
                .addEdge("polish", END)
                .compile();
    }

    /** The gate: plain code, not an LLM, decides whether the outline is good enough to continue. */
    static String gate(PromptChainState state) {
        if (countPoints(state.outline()) >= MIN_OUTLINE_POINTS) {
            return "valid";
        }
        return state.outlineAttempts() < MAX_OUTLINE_ATTEMPTS ? "retry" : "give_up";
    }

    static int countPoints(String outline) {
        return (int) outline.lines().map(String::strip).filter(line -> line.startsWith("- ")).count();
    }

    /** Canned replies so the sample runs without a model. */
    static ScriptedLlm placeholder() {
        return ScriptedLlm.create()
                .when("Create an outline", """
                        - What an agent is: an LLM in a loop with tools
                        - Why one big prompt is fragile
                        - How chaining small prompts helps""")
                .when("Write a draft", "An agent is an LLM in a loop... (draft that follows the outline)")
                .when("Polish this draft", "Why Small Prompts Win\n\nAn agent is an LLM running in a loop... (polished)");
    }

    public static void main(String[] args) throws Exception {
        String topic = args.length > 0 ? String.join(" ", args) : "why agents chain small prompts";

        PromptChainState result = build(LlmProvider.resolve(placeholder()))
                .invoke(Map.of(PromptChainState.TOPIC, topic))
                .orElseThrow();

        result.log().forEach(line -> System.out.println("[" + line + "]"));
        System.out.println();
        System.out.println(result.finalText().orElse(
                "Stopped: no usable outline after " + MAX_OUTLINE_ATTEMPTS + " attempt(s)."));
    }
}
