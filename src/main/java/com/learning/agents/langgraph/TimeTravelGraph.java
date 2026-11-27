package com.learning.agents.langgraph;

import com.learning.agents.llm.LlmProvider;
import com.learning.agents.llm.ScriptedLlm;
import org.bsc.langgraph4j.CompileConfig;
import org.bsc.langgraph4j.CompiledGraph;
import org.bsc.langgraph4j.GraphInput;
import org.bsc.langgraph4j.GraphStateException;
import org.bsc.langgraph4j.RunnableConfig;
import org.bsc.langgraph4j.StateGraph;
import org.bsc.langgraph4j.checkpoint.MemorySaver;
import org.bsc.langgraph4j.state.StateSnapshot;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import static org.bsc.langgraph4j.StateGraph.END;
import static org.bsc.langgraph4j.StateGraph.START;
import static org.bsc.langgraph4j.action.AsyncNodeAction.node_async;

/**
 * Time travel with checkpoints. With a checkpointer every step of a run is saved, so afterwards you can list the
 * history, go back to any step and <b>replay</b> from there (re-running the later nodes), or <b>fork</b>: change the
 * saved state at that step and run on from it, getting a different outcome while the original run stays in history.
 * This is how you debug "why did the agent do that?", retry from just before a bad step instead of from scratch, or
 * let a person correct the state mid-run.
 * <p>
 * Two LangGraph4j details: history needs {@code releaseThread(false)} at compile time, or a finished thread's
 * checkpoints are released; and {@code getStateHistory} lists snapshots newest first. Each snapshot's
 * {@code nextNodeId()} says which node would run next if you resumed from it.
 *
 * <pre>
 *   START → draft → polish → END          (each arrow is a saved checkpoint)
 *
 *   fork: go back to the checkpoint before "draft", set tone = "playful", resume → a different tagline
 * </pre>
 */
public class TimeTravelGraph {

    private final CompiledGraph<TaglineState> graph;
    private final RunnableConfig thread;

    /**
     * Builds the graph with an in-memory checkpointer, bound to one conversation thread.
     *
     * @param llm      maps a prompt to a model reply
     * @param threadId the conversation whose checkpoints this instance reads and writes
     */
    TimeTravelGraph(Function<String, String> llm, String threadId) throws GraphStateException {
        this.graph = new StateGraph<>(TaglineState.SCHEMA, TaglineState::new)
                .addNode("draft", node_async(state -> Map.of(TaglineState.DRAFT, llm.apply(
                        "Draft a %s tagline for %s.".formatted(state.tone(), state.product())).strip())))
                .addNode("polish", node_async(state -> Map.of(TaglineState.TAGLINE, llm.apply(
                        "Polish this tagline to at most eight words: " + state.draft()).strip())))
                .addEdge(START, "draft")
                .addEdge("draft", "polish")
                .addEdge("polish", END)
                .compile(CompileConfig.builder()
                        .checkpointSaver(new MemorySaver())
                        .releaseThread(false)   // keep the thread's checkpoints after the run, so history is available
                        .build());
        this.thread = RunnableConfig.builder().threadId(threadId).build();
    }

    /** Runs the graph from the start on this thread. */
    TaglineState run(String product) {
        return graph.invoke(GraphInput.args(Map.of(TaglineState.PRODUCT, product)), thread).orElseThrow();
    }

    /** Every saved step of this thread, oldest first (LangGraph4j returns them newest first). */
    List<StateSnapshot<TaglineState>> history() {
        List<StateSnapshot<TaglineState>> snapshots = new ArrayList<>(graph.getStateHistory(thread));
        Collections.reverse(snapshots);
        return snapshots;
    }

    /**
     * The most recent checkpoint saved just before {@code node} ran.
     *
     * @throws IllegalArgumentException if no checkpoint has {@code node} as its next node
     */
    StateSnapshot<TaglineState> checkpointBefore(String node) {
        return graph.getStateHistory(thread).stream()
                .filter(snapshot -> node.equals(snapshot.nextNodeId()))
                .findFirst()
                .orElseThrow(() -> new IllegalArgumentException("no checkpoint before node " + node));
    }

    /** Resumes from a saved checkpoint as it was, re-running every node after it. */
    TaglineState replay(String checkpointId) {
        return graph.invoke(GraphInput.resume(), at(checkpointId)).orElseThrow();
    }

    /** Changes the state saved at a checkpoint, then resumes from the changed copy; the original stays in history. */
    TaglineState fork(String checkpointId, Map<String, Object> changes) throws Exception {
        RunnableConfig forked = graph.updateState(at(checkpointId), changes);
        return graph.invoke(GraphInput.resume(), forked).orElseThrow();
    }

    private RunnableConfig at(String checkpointId) {
        return RunnableConfig.builder().threadId(thread.threadId().orElseThrow()).checkPointId(checkpointId).build();
    }

    /** Canned replies: one tagline per tone, so the fork visibly changes the result. */
    static ScriptedLlm placeholder() {
        return ScriptedLlm.create()
                .when("Draft a friendly", "Your notes, always with you, wherever you go")
                .when("Draft a playful", "Notes so handy, your brain gets jealous")
                .when(p -> p.startsWith("Polish") && p.contains("brain"), p -> "Notes so handy your brain gets jealous.")
                .when("Polish", "Your notes, always with you.");
    }

    public static void main(String[] args) throws Exception {
        String product = args.length > 0 ? String.join(" ", args) : "a note-taking app";
        var agent = new TimeTravelGraph(LlmProvider.resolve(placeholder()), "demo");

        System.out.println("Original run: " + agent.run(product).tagline().orElse("(none)"));

        System.out.println("\nHistory (oldest first):");
        for (var snapshot : agent.history()) {
            System.out.printf("  after %-9s next %-9s draft=\"%s\"%n",
                    snapshot.node(), snapshot.nextNodeId(), snapshot.state().draft());
        }

        String beforeDraft = agent.checkpointBefore("draft").checkpointId();
        System.out.println("\nReplay from before 'draft':  " + agent.replay(beforeDraft).tagline().orElse("(none)"));
        System.out.println("Fork with tone = \"playful\": " + agent.fork(beforeDraft, Map.of(TaglineState.TONE, "playful"))
                .tagline().orElse("(none)"));
        System.out.println("Checkpoints now in history: " + agent.history().size());
    }
}
