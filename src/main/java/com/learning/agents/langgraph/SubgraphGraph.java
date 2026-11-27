package com.learning.agents.langgraph;

import com.learning.agents.llm.LlmProvider;
import com.learning.agents.llm.ScriptedLlm;
import com.learning.agents.llm.TextLists;
import org.bsc.langgraph4j.CompiledGraph;
import org.bsc.langgraph4j.GraphInput;
import org.bsc.langgraph4j.GraphStateException;
import org.bsc.langgraph4j.RunnableConfig;
import org.bsc.langgraph4j.StateGraph;

import java.util.Map;
import java.util.function.Function;

import static org.bsc.langgraph4j.StateGraph.END;
import static org.bsc.langgraph4j.StateGraph.START;
import static org.bsc.langgraph4j.action.AsyncEdgeAction.edge_async;
import static org.bsc.langgraph4j.action.AsyncNodeAction.node_async;

/**
 * Subgraphs: a compiled graph used as a single node of another graph. The "research" team below is its own graph,
 * with a small loop that gathers notes until it has enough, and the parent "report" graph uses it like any other
 * step: outline → research → write. This is how larger agent systems are built from parts that can each be built and
 * tested on their own ({@link #research} is tested without the parent) and that hide their inner steps from the parent.
 * <p>
 * The parent and the subgraph share one state type ({@link ReportState}): the subgraph reads the outline the parent
 * wrote and writes notes and a summary the parent then uses.
 * <p>
 * LangGraph4j accepts a subgraph in two forms, and the difference matters. {@code addNode(id, StateGraph)} (used
 * here) inlines the subgraph's nodes into the parent, so they update the shared state exactly like parent nodes.
 * {@code addNode(id, CompiledGraph)} runs it as a separate graph and merges its <em>whole</em> final state back
 * afterwards, so any list the parent had already filled gets appended a second time.
 *
 * <pre>
 *   report:    START → outline → [ research ] → write → END
 *                                    │
 *   research:          START → gather ─┬─ fewer than 3 notes and rounds left → gather
 *                                      └─ enough → summarize → END
 * </pre>
 */
public class SubgraphGraph {

    static final int MIN_NOTES = 3;
    static final int MAX_ROUNDS = 3;

    /**
     * The research team as a standalone graph: gather notes (at most {@link #MAX_ROUNDS} rounds) until there are
     * {@link #MIN_NOTES}, then summarise them.
     */
    public static StateGraph<ReportState> research(Function<String, String> llm) throws GraphStateException {
        return new StateGraph<>(ReportState.SCHEMA, ReportState::new)
                .addNode("gather", node_async(state -> {
                    String reply = llm.apply("""
                            You are a researcher. Give one or two new facts, one per line, for this outline.
                            Already known:
                            %s
                            Outline: %s""".formatted(String.join("\n", state.notes()), state.outline()));
                    var facts = TextLists.items(reply, 2);
                    return Map.of(ReportState.NOTES, facts, ReportState.ROUNDS, state.rounds() + 1,
                                ReportState.RESEARCH_LOG, "gather round %d: +%d note(s)".formatted(state.rounds() + 1, facts.size()));
                }))
                .addNode("summarize", node_async(state -> Map.of(
                        ReportState.SUMMARY, llm.apply("Summarise these research notes in two sentences:\n"
                                + String.join("\n", state.notes())).strip(),
                        ReportState.RESEARCH_LOG, "summarize: done")))
                .addEdge(START, "gather")
                .addConditionalEdges("gather",
                        edge_async(SubgraphGraph::afterGather),
                        Map.of("gather", "gather", "summarize", "summarize"))
                .addEdge("summarize", END);
    }

    /**
     * The parent graph, with the research subgraph plugged in as the {@code research} node.
     *
     * @param llm maps a prompt to a model reply
     */
    public static CompiledGraph<ReportState> build(Function<String, String> llm) throws GraphStateException {
        return new StateGraph<>(ReportState.SCHEMA, ReportState::new)
                .addNode("outline", node_async(state -> Map.of(
                        ReportState.OUTLINE, llm.apply("Write a one-line outline for a short report on: " + state.topic()).strip(),
                        ReportState.LOG, "outline: done")))
                .addNode("research", research(llm))   // the whole research graph is one step here
                .addNode("write", node_async(state -> Map.of(
                        ReportState.REPORT, llm.apply("""
                                Write a three-sentence report.
                                Outline: %s
                                Research summary: %s""".formatted(state.outline(), state.summary())).strip(),
                        ReportState.LOG, "write: done")))
                .addEdge(START, "outline")
                .addEdge("outline", "research")
                .addEdge("research", "write")
                .addEdge("write", END)
                .compile();
    }

    /** Keeps gathering until {@link #MIN_NOTES} notes, but never for more than {@link #MAX_ROUNDS} rounds. */
    static String afterGather(ReportState state) {
        return state.notes().size() < MIN_NOTES && state.rounds() < MAX_ROUNDS ? "gather" : "summarize";
    }

    /** Runs the report graph once for {@code topic} and returns the final state. */
    static ReportState run(CompiledGraph<ReportState> graph, String topic) {
        return graph.invoke(GraphInput.args(Map.of(ReportState.TOPIC, topic)), RunnableConfig.builder().build()).orElseThrow();
    }

    /** Canned replies: the first research round finds two facts, the second finds one more. */
    static ScriptedLlm placeholder() {
        return ScriptedLlm.create()
                .when("one-line outline", "Why small open models are good enough for many agent tasks")
                .when(p -> p.startsWith("You are a researcher") && p.contains("8B models"),
                        p -> "Quantised models run on a laptop GPU")
                .when("You are a researcher", "8B models handle routing and extraction well\nThey struggle with long multi-step plans")
                .when("Summarise these research notes",
                        "Small open models handle routing and extraction and run on laptops. They are weaker at long plans.")
                .when("three-sentence report",
                        "Small open models are good enough for many agent tasks. They route and extract reliably on modest hardware. "
                                + "Keep long multi-step planning for larger models.");
    }

    public static void main(String[] args) throws Exception {
        String topic = args.length > 0 ? String.join(" ", args) : "small open models for agents";
        ReportState result = run(build(LlmProvider.resolve(placeholder())), topic);

        result.log().forEach(line -> System.out.println("[report] " + line));
        result.researchLog().forEach(line -> System.out.println("[research] " + line));
        System.out.println("\nNotes: " + result.notes());
        System.out.println("Report: " + result.report().orElse("(none)"));
    }
}
