package com.learning.agents.graph;

import com.learning.agents.llm.LlmProvider;
import com.learning.agents.llm.ScriptedLlm;
import org.bsc.langgraph4j.CompiledGraph;
import org.bsc.langgraph4j.GraphStateException;
import org.bsc.langgraph4j.StateGraph;

import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.bsc.langgraph4j.StateGraph.END;
import static org.bsc.langgraph4j.StateGraph.START;
import static org.bsc.langgraph4j.action.AsyncEdgeAction.edge_async;
import static org.bsc.langgraph4j.action.AsyncNodeAction.node_async;

/**
 * Orchestrator-workers: an orchestrator LLM call breaks the task into subtasks <em>at run time</em> (unlike prompt
 * chaining, where the steps are fixed in code), a worker handles each subtask with only the context it needs, and a
 * synthesizer merges the results. The plan is capped at {@link #MAX_SUBTASKS} so a verbose model can't make the
 * loop run away. Workers run one after another here to keep the sample simple; in production they are often run in
 * parallel because they don't depend on each other.
 *
 * <pre>
 *   START → orchestrator → worker ─┬─ subtasks left → worker
 *                            ↑     │
 *                            └─────┘
 *                                  └─ all done → synthesizer → END
 * </pre>
 */
public class OrchestratorWorkersGraph {

    static final int MAX_SUBTASKS = 4;

    /**
     * Builds the graph around a pluggable LLM so it can be tested without Ollama.
     *
     * @param llm maps a prompt to a model reply
     */
    public static CompiledGraph<OrchestratorState> build(Function<String, String> llm) throws GraphStateException {
        return new StateGraph<>(OrchestratorState.SCHEMA, OrchestratorState::new)
                .addNode("orchestrator", node_async(state -> {
                    String plan = llm.apply("""
                            You are an orchestrator. Break this task into 2 to %d independent subtasks.
                            Reply with one subtask per line and nothing else.
                            Task: %s""".formatted(MAX_SUBTASKS, state.task()));
                    return Map.of(OrchestratorState.SUBTASKS, parsePlan(plan), OrchestratorState.NEXT_INDEX, 0);
                }))
                .addNode("worker", node_async(state -> {
                    int i = state.nextIndex();
                    String subtask = state.subtasks().get(i);
                    // The worker sees the overall task for context, but is told to do only its own subtask.
                    String result = llm.apply("""
                            You are a worker. The overall task is: %s
                            Do only this subtask and reply in at most three sentences.
                            Subtask: %s""".formatted(state.task(), subtask));
                    return Map.of(OrchestratorState.RESULTS, result, OrchestratorState.NEXT_INDEX, i + 1);
                }))
                .addNode("synthesizer", node_async(state -> {
                    String report = llm.apply("""
                            Combine these subtask results into one coherent answer to the task.
                            Task: %s
                            Results:
                            %s""".formatted(state.task(), numbered(state.subtasks(), state.results())));
                    return Map.of(OrchestratorState.REPORT, report);
                }))
                .addEdge(START, "orchestrator")
                .addConditionalEdges("orchestrator",
                        edge_async(OrchestratorWorkersGraph::nextStep),
                        Map.of("worker", "worker", "synthesize", "synthesizer"))
                .addConditionalEdges("worker",
                        edge_async(OrchestratorWorkersGraph::nextStep),
                        Map.of("worker", "worker", "synthesize", "synthesizer"))
                .addEdge("synthesizer", END)
                .compile();
    }

    static String nextStep(OrchestratorState state) {
        return state.nextIndex() < state.subtasks().size() ? "worker" : "synthesize";
    }

    /**
     * Turns the orchestrator's reply into a clean list: drops blanks, list markers and preamble lines such as
     * "Here are the subtasks:", and keeps at most {@link #MAX_SUBTASKS}.
     */
    static List<String> parsePlan(String plan) {
        return plan.lines()
                .map(line -> line.strip().replaceFirst("^([-*•]|\\d+[.)])\\s*", ""))
                .filter(line -> !line.isBlank() && !line.endsWith(":"))
                .limit(MAX_SUBTASKS)
                .toList();
    }

    static String numbered(List<String> subtasks, List<String> results) {
        return IntStream.range(0, Math.min(subtasks.size(), results.size()))
                .mapToObj(i -> "%d. %s → %s".formatted(i + 1, subtasks.get(i), results.get(i)))
                .collect(Collectors.joining("\n"));
    }

    /** Canned replies so the sample runs without a model. */
    static ScriptedLlm placeholder() {
        return ScriptedLlm.create()
                .when("You are an orchestrator", """
                        1. List the costs of running a model locally
                        2. List the costs of a hosted API
                        3. Compare privacy trade-offs""")
                .when("Subtask: List the costs of running", "Hardware and electricity, but no per-token fees.")
                .when("Subtask: List the costs of a hosted", "Pay per token; no hardware to buy.")
                .when("Subtask: Compare privacy", "Local keeps data on your network; hosted sends it out.")
                .when("Combine these subtask results",
                        "Local models trade upfront hardware cost for privacy and no per-token fees.");
    }

    public static void main(String[] args) throws Exception {
        String task = args.length > 0
                ? String.join(" ", args)
                : "Should a small team run LLMs locally or use a hosted API?";

        OrchestratorState result = build(LlmProvider.resolve(placeholder()))
                .invoke(Map.of(OrchestratorState.TASK, task))
                .orElseThrow();

        System.out.println("Plan and worker results:");
        System.out.println(numbered(result.subtasks(), result.results()));
        System.out.println();
        System.out.println("Report: " + result.report().orElse("(none)"));
    }
}
