package com.learning.agents.graph;

import com.learning.agents.OllamaConfig;
import dev.langchain4j.model.chat.ChatModel;
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
 * A LangGraph4j workflow with a loop: a "writer" node drafts text, a "reviewer" node critiques it,
 * and a conditional edge sends the draft back for revision until it is approved or
 * {@link #MAX_REVISIONS} is reached.
 *
 * <pre>
 *   START → writer → reviewer ─┬─ approved / limit reached → END
 *              ↑               │
 *              └──── revise ───┘
 * </pre>
 */
public class WriterReviewerGraph {

    static final int MAX_REVISIONS = 2;
    static final String APPROVED = "APPROVED";

    /**
     * Builds the graph around two pluggable functions so it can be tested without an LLM.
     *
     * @param llm maps a prompt to a model reply
     */
    public static CompiledGraph<WriterState> build(Function<String, String> llm) throws GraphStateException {
        return new StateGraph<>(WriterState.SCHEMA, WriterState::new)
                .addNode("writer", node_async(state -> {
                    String prompt = state.feedback()
                            .map(fb -> "Rewrite this short paragraph about \"%s\" using the feedback.%nDraft:%n%s%nFeedback:%n%s"
                                    .formatted(state.topic(), state.draft().orElse(""), fb))
                            .orElse("Write a short paragraph (max 80 words) about \"%s\".".formatted(state.topic()));
                    String draft = llm.apply(prompt);
                    return Map.of(
                            WriterState.DRAFT, draft,
                            WriterState.REVISIONS, state.revisions() + 1,
                            WriterState.LOG, "writer: produced draft #" + (state.revisions() + 1));
                }))
                .addNode("reviewer", node_async(state -> {
                    String prompt = ("You are a strict editor. If the paragraph below is clear, accurate and under 80 words, "
                            + "reply with exactly %s. Otherwise reply with one or two sentences of concrete feedback.%n%n%s")
                            .formatted(APPROVED, state.draft().orElse(""));
                    String feedback = llm.apply(prompt).trim();
                    return Map.of(
                            WriterState.FEEDBACK, feedback,
                            WriterState.LOG, "reviewer: " + feedback);
                }))
                .addEdge(START, "writer")
                .addEdge("writer", "reviewer")
                .addConditionalEdges("reviewer",
                        edge_async(state -> isDone(state) ? "done" : "revise"),
                        Map.of("done", END, "revise", "writer"))
                .compile();
    }

    static boolean isDone(WriterState state) {
        return state.feedback().map(fb -> fb.toUpperCase().startsWith(APPROVED)).orElse(false)
                || state.revisions() >= MAX_REVISIONS;
    }

    public static void main(String[] args) throws Exception {
        ChatModel model = OllamaConfig.chatModel();
        String topic = args.length > 0 ? String.join(" ", args) : "why agents need memory";

        WriterState result = build(model::chat)
                .invoke(Map.of(WriterState.TOPIC, topic))
                .orElseThrow();

        result.log().forEach(entry -> System.out.println("- " + entry));
        System.out.println();
        System.out.println("Final draft (" + result.revisions() + " revision(s)):");
        System.out.println(result.draft().orElse("<none>"));
    }
}
