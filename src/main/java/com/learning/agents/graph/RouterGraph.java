package com.learning.agents.graph;

import com.learning.agents.llm.LlmProvider;
import com.learning.agents.llm.ScriptedLlm;
import org.bsc.langgraph4j.CompiledGraph;
import org.bsc.langgraph4j.GraphStateException;
import org.bsc.langgraph4j.StateGraph;

import java.util.Locale;
import java.util.Map;
import java.util.function.Function;

import static org.bsc.langgraph4j.StateGraph.END;
import static org.bsc.langgraph4j.StateGraph.START;
import static org.bsc.langgraph4j.action.AsyncEdgeAction.edge_async;
import static org.bsc.langgraph4j.action.AsyncNodeAction.node_async;

/**
 * Routing: a cheap classification call decides which specialist handles the request, and each specialist has its
 * own focused prompt. The classifier's free-text reply is normalised by code, and anything unexpected falls back to
 * the {@code general} route, so a confused model can never send the request to a node that doesn't exist.
 *
 * <pre>
 *                     ┌─ billing ──┐
 *   START → classify ─┼─ technical ┼→ END
 *                     └─ general ──┘
 * </pre>
 */
public class RouterGraph {

    static final String BILLING = "billing";
    static final String TECHNICAL = "technical";
    static final String GENERAL = "general";

    /** Each route's specialist instructions. Edit these to see how much the prompt shapes the answer. */
    static final Map<String, String> SPECIALISTS = Map.of(
            BILLING, "You are a billing specialist. Be precise about charges, refunds and invoices.",
            TECHNICAL, "You are a senior support engineer. Give numbered troubleshooting steps.",
            GENERAL, "You are a friendly front-desk assistant. Answer briefly and politely."
    );

    /**
     * Builds the router around a pluggable LLM so it can be tested without Ollama.
     *
     * @param llm maps a prompt to a model reply
     */
    public static CompiledGraph<RouterState> build(Function<String, String> llm) throws GraphStateException {
        var graph = new StateGraph<>(RouterState.SCHEMA, RouterState::new)
                .addNode("classify", node_async(state -> {
                    String reply = llm.apply("""
                            Classify the customer message into exactly one category: billing, technical or general.
                            Reply with the single category word only.
                            Message: %s""".formatted(state.query()));
                    return Map.of(RouterState.ROUTE, normalizeRoute(reply));
                }));
        for (var specialist : SPECIALISTS.entrySet()) {
            graph.addNode(specialist.getKey(), node_async(state -> Map.of(RouterState.ANSWER, llm.apply("""
                    %s
                    Customer message: %s""".formatted(specialist.getValue(), state.query())))));
            graph.addEdge(specialist.getKey(), END);
        }
        return graph
                .addEdge(START, "classify")
                .addConditionalEdges("classify",
                        edge_async(RouterState::route),
                        Map.of(BILLING, BILLING, TECHNICAL, TECHNICAL, GENERAL, GENERAL))
                .compile();
    }

    /** Maps a free-text classifier reply to a known route; anything unrecognised becomes {@code general}. */
    static String normalizeRoute(String reply) {
        String text = reply.toLowerCase(Locale.ROOT);
        if (text.contains(BILLING)) {
            return BILLING;
        }
        if (text.contains(TECHNICAL)) {
            return TECHNICAL;
        }
        return GENERAL;
    }

    /** Canned replies so the sample runs without a model. */
    static ScriptedLlm placeholder() {
        return ScriptedLlm.create()
                .when(prompt -> prompt.startsWith("Classify") && prompt.toLowerCase(Locale.ROOT).contains("charged"),
                        prompt -> "Billing")
                .when(prompt -> prompt.startsWith("Classify") && prompt.toLowerCase(Locale.ROOT).contains("error"),
                        prompt -> "technical")
                .when("Classify", "general")
                .when("billing specialist", "I can see a duplicate charge; a refund has been requested.")
                .when("support engineer", "1. Restart the app. 2. Clear the cache. 3. Reinstall if it persists.")
                .otherwise("Happy to help! Our office is open 9 to 5.");
    }

    public static void main(String[] args) throws Exception {
        var router = build(LlmProvider.resolve(placeholder()));
        var queries = args.length > 0
                ? new String[]{String.join(" ", args)}
                : new String[]{"I was charged twice this month", "The app shows error 500 on login", "What are your opening hours?"};

        for (String query : queries) {
            RouterState result = router.invoke(Map.of(RouterState.QUERY, query)).orElseThrow();
            System.out.printf("%-40s → %-9s → %s%n", query, result.route(), result.answer().orElse(""));
        }
    }
}
