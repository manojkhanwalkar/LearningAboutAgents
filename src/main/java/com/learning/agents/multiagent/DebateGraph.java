package com.learning.agents.multiagent;

import com.learning.agents.llm.LlmProvider;
import com.learning.agents.llm.PromptText;
import com.learning.agents.llm.Replies;
import com.learning.agents.llm.ScriptedLlm;
import org.bsc.langgraph4j.CompiledGraph;
import org.bsc.langgraph4j.GraphInput;
import org.bsc.langgraph4j.GraphStateException;
import org.bsc.langgraph4j.RunnableConfig;
import org.bsc.langgraph4j.StateGraph;

import java.util.Locale;
import java.util.Map;
import java.util.function.Function;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.bsc.langgraph4j.StateGraph.END;
import static org.bsc.langgraph4j.StateGraph.START;
import static org.bsc.langgraph4j.action.AsyncEdgeAction.edge_async;
import static org.bsc.langgraph4j.action.AsyncNodeAction.node_async;

/**
 * Debate (Du et al., 2023): two agents argue opposite sides of a question for {@link #ROUNDS} rounds, each seeing and
 * rebutting the other's arguments, then a separate judge reads the whole transcript and decides. Arguing both sides
 * surfaces considerations a single answer skips, and a judge that didn't write either side is less attached to one.
 * <p>
 * The judge's verdict is parsed strictly ({@code WINNER: PRO} or {@code WINNER: CON} on its own line); anything else is
 * recorded as {@code UNDECIDED} rather than guessed.
 *
 * <pre>
 *   START → pro → con ─┬─ rounds left → pro
 *                      └─ done → judge → END
 * </pre>
 */
public class DebateGraph {

    static final int ROUNDS = 2;

    /** Applied to the single line after {@code WINNER:}: the side, then optionally a reason on the same line. */
    private static final Pattern SIDE = Pattern.compile("^\\W*(PRO|CON)\\b\\W*(.*)$", Pattern.CASE_INSENSITIVE);

    /**
     * Builds the graph around a pluggable LLM so it can be tested without Ollama.
     *
     * @param llm maps a prompt to a model reply
     */
    public static CompiledGraph<DebateState> build(Function<String, String> llm) throws GraphStateException {
        return new StateGraph<>(DebateState.SCHEMA, DebateState::new)
                .addNode("pro", node_async(state -> Map.of(DebateState.TRANSCRIPT,
                        String.format(Locale.ROOT, "PRO (round %d): %s", state.round() + 1, argue(llm, "FOR", state)))))
                .addNode("con", node_async(state -> Map.of(
                        DebateState.TRANSCRIPT, String.format(Locale.ROOT, "CON (round %d): %s", state.round() + 1, argue(llm, "AGAINST", state)),
                        DebateState.ROUND, state.round() + 1)))
                .addNode("judge", node_async(state -> {
                    String verdict = llm.apply("""
                            You are an impartial judge. Read the debate and decide which side argued better.
                            Reply with "WINNER: PRO" or "WINNER: CON" on the first line, then one sentence of reasoning.
                            Motion: %s
                            Debate:
                            %s""".formatted(motion(state), PromptText.escape(String.join("\n", state.transcript()))));
                    var line = Replies.lineAfter(verdict, "WINNER:");
                    var m = line.map(SIDE::matcher).filter(Matcher::find);
                    if (m.isEmpty()) {
                        return Map.of(DebateState.WINNER, "UNDECIDED", DebateState.REASON, "the judge's verdict was unreadable");
                    }
                    // The reason is the rest of the verdict line, or else the text on the lines after it.
                    String afterLine = Replies.after(verdict, "WINNER:").orElse("").substring(line.get().length()).strip();
                    String reason = !m.get().group(2).isBlank() ? m.get().group(2).strip() : afterLine;
                    return Map.of(DebateState.WINNER, m.get().group(1).toUpperCase(Locale.ROOT), DebateState.REASON, reason);
                }))
                .addEdge(START, "pro")
                .addEdge("pro", "con")
                .addConditionalEdges("con",
                        edge_async(state -> state.round() < ROUNDS ? "pro" : "judge"),
                        Map.of("pro", "pro", "judge", "judge"))
                .addEdge("judge", END)
                .compile();
    }

    private static String argue(Function<String, String> llm, String side, DebateState state) {
        return llm.apply("""
                You are debating %s the motion. Make one new point in at most two sentences, and rebut the other side's
                latest point if there is one.
                Motion: %s
                Debate so far:
                %s""".formatted(side, motion(state),
                state.transcript().isEmpty() ? "(you speak first)" : PromptText.escape(String.join("\n", state.transcript())))).strip();
    }

    private static String motion(DebateState state) {
        return "<motion>" + PromptText.escape(state.motion()) + "</motion>";
    }

    static DebateState run(CompiledGraph<DebateState> graph, String motion) {
        return graph.invoke(GraphInput.args(Map.of(DebateState.MOTION, motion)), RunnableConfig.builder().build()).orElseThrow();
    }

    /** Canned replies: two rounds each way, then a verdict for CON. */
    static ScriptedLlm placeholder() {
        return ScriptedLlm.create()
                .when(p -> p.startsWith("You are debating FOR") && p.contains("CON (round 1)"),
                        p -> "Ongoing help is free with open-source communities. Lock-in costs more than support.")
                .when(p -> p.startsWith("You are debating FOR"),
                        p -> "Open-source models avoid vendor lock-in and keep data in-house.")
                .when(p -> p.startsWith("You are debating AGAINST") && p.contains("PRO (round 2)"),
                        p -> "Community help has no deadline; a startup can't wait days for a fix.")
                .when(p -> p.startsWith("You are debating AGAINST"),
                        p -> "A small startup lacks the staff to run and patch models; managed APIs ship faster.")
                .when("impartial judge", "WINNER: CON\nCON tied its points to a startup's real constraint: limited staff and time.");
    }

    public static void main(String[] args) throws Exception {
        String motion = args.length > 0 ? String.join(" ", args) : "A five-person startup should self-host open-source LLMs.";
        DebateState result = run(build(LlmProvider.resolve(placeholder())), motion);

        System.out.println("Motion: " + motion);
        result.transcript().forEach(line -> System.out.println("  " + line));
        System.out.println("Winner: " + result.winner().orElse("?") + " (" + result.reason() + ")");
    }
}
