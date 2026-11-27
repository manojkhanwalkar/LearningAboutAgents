package com.learning.agents.graph;

import com.learning.agents.llm.LlmProvider;
import com.learning.agents.llm.ScriptedLlm;
import org.bsc.langgraph4j.CompileConfig;
import org.bsc.langgraph4j.CompiledGraph;
import org.bsc.langgraph4j.GraphInput;
import org.bsc.langgraph4j.GraphStateException;
import org.bsc.langgraph4j.RunnableConfig;
import org.bsc.langgraph4j.StateGraph;
import org.bsc.langgraph4j.checkpoint.MemorySaver;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.function.Function;
import java.util.function.Predicate;

import static org.bsc.langgraph4j.StateGraph.END;
import static org.bsc.langgraph4j.StateGraph.START;
import static org.bsc.langgraph4j.action.AsyncEdgeAction.edge_async;
import static org.bsc.langgraph4j.action.AsyncNodeAction.node_async;

/**
 * Human in the loop: the agent proposes an action, then the graph <em>pauses</em> before anything happens so a
 * person can approve or reject it. Pausing uses two LangGraph4j features: a checkpointer ({@link MemorySaver}) that
 * saves the state after every node, and {@code interruptBefore("human_review")}, which stops the run there. The
 * caller records the decision with {@code updateState} and resumes from the checkpoint with {@link GraphInput#resume()}.
 * <p>
 * Use this for any irreversible or outward-facing action (sending email, spending money, deleting data). The state
 * defaults to "not approved", so a missing decision never triggers the action.
 *
 * <pre>
 *   START → propose → ⏸ human_review ─┬─ approved → execute → END
 *                                     └─ rejected → cancel ──→ END
 * </pre>
 */
public class HumanApprovalGraph {

    static final String REVIEW_NODE = "human_review";

    /**
     * Builds the graph around a pluggable LLM. The returned graph pauses before {@value #REVIEW_NODE}.
     *
     * @param llm maps a prompt to a model reply
     */
    public static CompiledGraph<ApprovalState> build(Function<String, String> llm) throws GraphStateException {
        return new StateGraph<>(ApprovalState.SCHEMA, ApprovalState::new)
                .addNode("propose", node_async(state -> Map.of(ApprovalState.PROPOSAL, llm.apply("""
                        You are an operations agent. Propose exactly one concrete action that fulfils the request.
                        Reply with a single line starting with "ACTION: ".
                        Request: %s""".formatted(state.request())).strip())))
                // Does nothing itself; it exists so the graph has a place to pause before the decision is read.
                .addNode(REVIEW_NODE, node_async(state -> Map.of()))
                .addNode("execute", node_async(state -> Map.of(ApprovalState.RESULT, llm.apply("""
                        The following action was approved by a human. Report, in one sentence, that it was done.
                        %s""".formatted(state.proposal())))))
                .addNode("cancel", node_async(state -> Map.of(ApprovalState.RESULT, "Cancelled by reviewer: " + state.proposal())))
                .addEdge(START, "propose")
                .addEdge("propose", REVIEW_NODE)
                .addConditionalEdges(REVIEW_NODE,
                        edge_async(state -> state.approved() ? "execute" : "cancel"),
                        Map.of("execute", "execute", "cancel", "cancel"))
                .addEdge("execute", END)
                .addEdge("cancel", END)
                .compile(CompileConfig.builder()
                        .checkpointSaver(new MemorySaver())
                        .interruptBefore(REVIEW_NODE)
                        .build());
    }

    /**
     * Runs the whole approval flow: start, pause, ask {@code reviewer}, record the answer, resume.
     *
     * @param reviewer shown the proposal; returns {@code true} to approve
     * @return the final state
     */
    static ApprovalState run(Function<String, String> llm, String request, Predicate<String> reviewer) throws Exception {
        var graph = build(llm);
        // Each conversation gets its own thread id, so its checkpoints are kept apart from other runs.
        var config = RunnableConfig.builder().threadId(UUID.randomUUID().toString()).build();

        ApprovalState paused = graph.invoke(Map.of(ApprovalState.REQUEST, request), config).orElseThrow();
        boolean approved = reviewer.test(paused.proposal());

        var resumeConfig = graph.updateState(config, Map.of(ApprovalState.APPROVED, approved));
        return graph.invoke(GraphInput.resume(), resumeConfig).orElseThrow();
    }

    /** Asks on the console; anything other than {@code y}/{@code yes}, including end of input, is a rejection. */
    static Predicate<String> consoleReviewer(BufferedReader in) {
        return proposal -> {
            System.out.println("The agent wants to: " + proposal);
            System.out.print("Approve? [y/N] ");
            System.out.flush();
            try {
                String answer = in.readLine();
                return answer != null && Set.of("y", "yes").contains(answer.strip().toLowerCase(Locale.ROOT));
            } catch (IOException e) {
                return false;
            }
        };
    }

    /** Canned replies so the sample runs without a model. */
    static ScriptedLlm placeholder() {
        return ScriptedLlm.create()
                .when("operations agent", "ACTION: Restart the staging web server (web-staging-1).")
                .when("was approved by a human", "Done: web-staging-1 was restarted.");
    }

    public static void main(String[] args) throws Exception {
        String request = args.length > 0 ? String.join(" ", args) : "The staging site is slow, fix it.";
        var in = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));

        ApprovalState result = run(LlmProvider.resolve(placeholder()), request, consoleReviewer(in));

        System.out.println(result.result().orElse("(no result)"));
    }
}
