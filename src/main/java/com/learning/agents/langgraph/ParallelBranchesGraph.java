package com.learning.agents.langgraph;

import com.learning.agents.llm.LlmProvider;
import com.learning.agents.llm.ScriptedLlm;
import org.bsc.langgraph4j.CompiledGraph;
import org.bsc.langgraph4j.GraphInput;
import org.bsc.langgraph4j.GraphStateException;
import org.bsc.langgraph4j.RunnableConfig;
import org.bsc.langgraph4j.StateGraph;

import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import java.util.function.Function;

import static org.bsc.langgraph4j.StateGraph.END;
import static org.bsc.langgraph4j.StateGraph.START;
import static org.bsc.langgraph4j.action.AsyncNodeAction.node_async;

/**
 * Parallel branches (fan-out / fan-in) in LangGraph4j: several edges leave the same node, so their targets form a
 * parallel step, and all of them lead into one join node that runs once every branch has finished. Each analyst here
 * looks at the question from one angle, independently, so there is no reason to wait for one before starting the next.
 * <p>
 * The catch: <b>by default LangGraph4j runs the branches one after another on a single thread</b>. They only overlap
 * when you register an executor for the node they fan out from, with
 * {@code RunnableConfig.builder().addParallelNodeExecutor(START, executor)}. {@link #main} runs both ways and prints
 * the time each takes. Branch results are merged through an appender channel ({@link ParallelState#VIEWS}), so the
 * branches never overwrite each other.
 *
 * <pre>
 *          ┌─ optimist ──┐
 *   START ─┼─ skeptic ───┼─→ judge → END
 *          └─ economist ─┘
 * </pre>
 */
public class ParallelBranchesGraph {

    /** The analysts: a node name and the angle it takes. Add one here to add a branch. */
    static final Map<String, String> ANALYSTS = Map.of(
            "optimist", "the strongest argument FOR",
            "skeptic", "the strongest argument AGAINST",
            "economist", "the main cost or benefit in money and time");

    /**
     * Builds the graph around a pluggable LLM so it can be tested without Ollama.
     *
     * @param llm maps a prompt to a model reply; called from several threads at once when run in parallel
     */
    public static CompiledGraph<ParallelState> build(Function<String, String> llm) throws GraphStateException {
        var graph = new StateGraph<>(ParallelState.SCHEMA, ParallelState::new);
        for (var analyst : ANALYSTS.entrySet()) {
            graph.addNode(analyst.getKey(), node_async(state -> Map.of(ParallelState.VIEWS,
                    analyst.getKey() + ": " + llm.apply("""
                            In one sentence, give %s.
                            Question: %s""".formatted(analyst.getValue(), state.question())).strip())));
            graph.addEdge(START, analyst.getKey());   // several edges from START = one parallel step
            graph.addEdge(analyst.getKey(), "judge");  // all branches join here
        }
        return graph
                .addNode("judge", node_async(state -> Map.of(ParallelState.VERDICT, llm.apply("""
                        Weigh these views and give a one-sentence verdict.
                        Question: %s
                        Views:
                        %s""".formatted(state.question(), String.join("\n", state.views().stream().sorted().toList()))))))
                .addEdge("judge", END)
                .compile();
    }

    /**
     * Runs the graph once.
     *
     * @param parallel the executor that runs the branches concurrently, or {@code null} for LangGraph4j's default
     *                 (one after another)
     */
    static ParallelState run(CompiledGraph<ParallelState> graph, String question, Executor parallel) {
        var config = RunnableConfig.builder();
        if (parallel != null) {
            config.addParallelNodeExecutor(START, parallel);
        }
        return graph.invoke(GraphInput.args(Map.of(ParallelState.QUESTION, question)), config.build()).orElseThrow();
    }

    /** Wraps an LLM so each call takes at least {@code latency}, like a real model would. */
    static Function<String, String> withLatency(Function<String, String> llm, Duration latency) {
        return prompt -> {
            try {
                Thread.sleep(latency);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("interrupted", e);
            }
            return llm.apply(prompt);
        };
    }

    /** Canned replies so the sample runs without a model. */
    static ScriptedLlm placeholder() {
        return ScriptedLlm.create()
                .when("argument FOR", "Local models keep data private and cost nothing per call.")
                .when("argument AGAINST", "Local hardware is slower and needs maintenance.")
                .when("money and time", "A one-off GPU purchase pays for itself within months at high volume.")
                .when("Weigh these views", "Run locally if volume is high and data is sensitive; otherwise start hosted.");
    }

    /** Times the same graph run one branch after another and with a virtual-thread executor, and prints both timings. */
    public static void main(String[] args) throws Exception {
        String question = args.length > 0 ? String.join(" ", args) : "Should our team run its LLM locally?";
        // The placeholder answers instantly, so give it a model-like delay to make the difference visible.
        Function<String, String> llm = LlmProvider.resolve(withLatency(placeholder(), Duration.ofMillis(300)));
        var graph = build(llm);

        long start = System.nanoTime();
        run(graph, question, null);
        long sequentialMs = (System.nanoTime() - start) / 1_000_000;

        ParallelState result;
        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            start = System.nanoTime();
            result = run(graph, question, executor);
        }
        long parallelMs = (System.nanoTime() - start) / 1_000_000;

        System.out.println("Question: " + question);
        List.copyOf(result.views()).stream().sorted().forEach(view -> System.out.println("  " + view));
        System.out.println("Verdict: " + result.verdict().orElse("(none)"));
        System.out.printf("%nDefault (branches one after another): %d ms%nWith a parallel executor: %d ms%n",
                sequentialMs, parallelMs);
    }
}
