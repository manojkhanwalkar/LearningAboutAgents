package com.learning.agents.graph;

import com.learning.agents.llm.LlmProvider;
import com.learning.agents.llm.ScriptedLlm;
import org.bsc.langgraph4j.CompiledGraph;
import org.bsc.langgraph4j.GraphStateException;
import org.bsc.langgraph4j.StateGraph;

import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.function.Function;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import static org.bsc.langgraph4j.StateGraph.END;
import static org.bsc.langgraph4j.StateGraph.START;
import static org.bsc.langgraph4j.action.AsyncEdgeAction.edge_async;
import static org.bsc.langgraph4j.action.AsyncNodeAction.node_async;

/**
 * Multi-agent supervisor: several specialist agents (each just a persona prompt here) share a scratchpad, and a
 * supervisor LLM reads the scratchpad and decides who works next, or that the job is finished. Control always
 * returns to the supervisor, which is what makes this different from a fixed pipeline. {@link #MAX_TURNS} stops the
 * loop if the supervisor never says {@code FINISH}.
 *
 * <pre>
 *            ┌──────────── researcher ←┐
 *            ↓                         │
 *   START → supervisor ─ NEXT: agent ──┼─ writer
 *            ↑  │                      │
 *            │  └─ FINISH / limit → END└─ critic
 *            └────────────────────────────┘ (every agent reports back)
 * </pre>
 */
public class SupervisorGraph {

    static final int MAX_TURNS = 6;
    static final String FINISH = "FINISH";

    /** A worker agent: a node name and the persona prompt that makes it a specialist. */
    record Agent(String name, String persona) {
    }

    /** The team, in the order the supervisor sees it. Add an entry here to add an agent. */
    static final List<Agent> AGENTS = List.of(
            new Agent("researcher", "You are a researcher. List 2-3 key facts relevant to the request. Do not write prose."),
            new Agent("writer", "You are a writer. Using the facts gathered so far, write a short answer (max 80 words)."),
            new Agent("critic", "You are a critic. Point out one weakness in the latest draft, or reply LGTM if there is none.")
    );

    private static final Set<String> AGENT_NAMES = AGENTS.stream().map(Agent::name).collect(Collectors.toUnmodifiableSet());

    private static final Pattern WORD = Pattern.compile("\\w+");

    /**
     * Builds the team around a pluggable LLM so it can be tested without Ollama.
     *
     * @param llm maps a prompt to a model reply
     */
    public static CompiledGraph<SupervisorState> build(Function<String, String> llm) throws GraphStateException {
        var graph = new StateGraph<>(SupervisorState.SCHEMA, SupervisorState::new)
                .addNode("supervisor", node_async(state -> {
                    String reply = llm.apply("""
                            You are a supervisor managing these agents: %s.
                            Read the work so far and decide who acts next.
                            Reply with "NEXT: <agent>" or, when the request is fully answered, "%s".
                            Request: %s
                            Work so far:
                            %s""".formatted(AGENTS.stream().map(Agent::name).collect(Collectors.joining(", ")), FINISH, state.request(),
                            state.work().isEmpty() ? "(nothing yet)" : String.join("\n", state.work())));
                    return Map.of(
                            SupervisorState.NEXT, parseNext(reply).orElse(FINISH),
                            SupervisorState.TURNS, state.turns() + 1);
                }));

        Map<String, String> routes = new HashMap<>();
        for (Agent agent : AGENTS) {
            String name = agent.name();
            graph.addNode(name, node_async(state -> {
                String output = llm.apply("""
                        %s
                        Request: %s
                        Work so far:
                        %s""".formatted(agent.persona(), state.request(), String.join("\n", state.work())));
                return Map.of(SupervisorState.WORK, name + ": " + output.strip());
            }));
            graph.addEdge(name, "supervisor");
            routes.put(name, name);
        }
        routes.put(FINISH, END);

        return graph
                .addEdge(START, "supervisor")
                .addConditionalEdges("supervisor", edge_async(SupervisorGraph::route), routes)
                .compile();
    }

    static String route(SupervisorState state) {
        return state.turns() >= MAX_TURNS ? FINISH : state.next();
    }

    /**
     * Reads "NEXT: agent" from the supervisor's reply: the first known agent name after {@code NEXT:}, so
     * "NEXT: the writer" and "**NEXT:** writer" both work. Empty if there is no {@code NEXT:} or no known agent.
     */
    static Optional<String> parseNext(String reply) {
        String lower = reply.toLowerCase(Locale.ROOT);
        int at = lower.indexOf("next:");
        if (at < 0) {
            return Optional.empty();
        }
        var words = WORD.matcher(lower.substring(at + "next:".length()));
        while (words.find()) {
            if (AGENT_NAMES.contains(words.group())) {
                return Optional.of(words.group());
            }
        }
        return Optional.empty();
    }

    /** The writer's latest output, which is the team's answer. */
    static Optional<String> latestDraft(List<String> work) {
        return work.reversed().stream()
                .filter(entry -> entry.startsWith("writer: "))
                .map(entry -> entry.substring("writer: ".length()))
                .findFirst();
    }

    /** Canned replies so the sample runs without a model: research → write → critique → finish. */
    static ScriptedLlm placeholder() {
        return ScriptedLlm.create()
                .when(prompt -> prompt.startsWith("You are a supervisor") && prompt.contains("critic: "),
                        prompt -> FINISH)
                .when(prompt -> prompt.startsWith("You are a supervisor") && prompt.contains("writer: "),
                        prompt -> "NEXT: critic")
                .when(prompt -> prompt.startsWith("You are a supervisor") && prompt.contains("researcher: "),
                        prompt -> "NEXT: writer")
                .when("You are a supervisor", "NEXT: researcher")
                .when("You are a researcher", "- Gemma is an open-weights model family\n- Ollama serves models locally")
                .when("You are a writer", "You can run open models such as Gemma on your own machine with Ollama.")
                .when("You are a critic", "LGTM");
    }

    public static void main(String[] args) throws Exception {
        String request = args.length > 0 ? String.join(" ", args) : "Explain how to run Gemma locally.";

        SupervisorState result = build(LlmProvider.resolve(placeholder()))
                .invoke(Map.of(SupervisorState.REQUEST, request))
                .orElseThrow();

        result.work().forEach(entry -> System.out.println(entry + "\n"));
        System.out.println("Supervisor turns: " + result.turns());
        System.out.println("Answer: " + latestDraft(result.work()).orElse("(the writer never ran)"));
    }
}
