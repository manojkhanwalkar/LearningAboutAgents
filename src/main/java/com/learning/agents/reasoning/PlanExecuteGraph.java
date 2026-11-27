package com.learning.agents.reasoning;

import com.learning.agents.llm.LlmProvider;
import com.learning.agents.llm.ScriptedLlm;
import com.learning.agents.llm.TextLists;
import org.bsc.langgraph4j.CompiledGraph;
import org.bsc.langgraph4j.GraphStateException;
import org.bsc.langgraph4j.StateGraph;

import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.regex.Pattern;

import static org.bsc.langgraph4j.StateGraph.END;
import static org.bsc.langgraph4j.StateGraph.START;
import static org.bsc.langgraph4j.action.AsyncEdgeAction.edge_async;
import static org.bsc.langgraph4j.action.AsyncNodeAction.node_async;

/**
 * Plan-and-execute with replanning: a planner writes a step-by-step plan, an executor does <em>one</em> step, and a
 * replanner looks at what that step revealed and rewrites the rest of the plan, or declares the goal met with
 * {@code FINAL:}. Unlike {@code graph.OrchestratorWorkersGraph}, which plans once, the plan here changes as the agent
 * learns. {@link #MAX_STEPS} caps the run, and a {@code finish} node always writes an answer from what was done.
 *
 * <pre>
 *   START → planner → executor → replanner ─┬─ FINAL: answer ─────────────→ END
 *                        ↑                  ├─ new plan → executor
 *                        └──────────────────┘
 *                                           └─ plan empty / step limit → finish → END
 * </pre>
 */
public class PlanExecuteGraph {

    static final int MAX_STEPS = 6;
    static final int MAX_PLAN_STEPS = 5;

    private static final Pattern FINAL = Pattern.compile("(?is)^\\s*FINAL:\\s*(.+)$");

    /**
     * Builds the graph around a pluggable LLM so it can be tested without Ollama.
     *
     * @param llm maps a prompt to a model reply
     */
    public static CompiledGraph<PlanExecuteState> build(Function<String, String> llm) throws GraphStateException {
        return new StateGraph<>(PlanExecuteState.SCHEMA, PlanExecuteState::new)
                .addNode("planner", node_async(state -> {
                    List<String> plan = TextLists.items(llm.apply("""
                            You are a planner. Write a short numbered plan (at most %d steps) to reach the goal.
                            Goal: %s""".formatted(MAX_PLAN_STEPS, state.goal())), MAX_PLAN_STEPS);
                    return Map.of(PlanExecuteState.PLAN, plan, PlanExecuteState.PLAN_HISTORY, String.join(" | ", plan));
                }))
                .addNode("executor", node_async(state -> {
                    String step = state.plan().getFirst();
                    String result = llm.apply("""
                            You are an executor. Do exactly this one step and report the result in one sentence.
                            Goal: %s
                            Done so far:
                            %s
                            Step: %s""".formatted(state.goal(), done(state), step)).strip();
                    return Map.of(PlanExecuteState.PAST_STEPS, step + " → " + result,
                            PlanExecuteState.STEPS, state.steps() + 1);
                }))
                .addNode("replanner", node_async(state -> {
                    String reply = llm.apply("""
                            You are a replanner. Given the goal, the steps done and the old plan, either
                            reply "FINAL: <answer>" if the goal is met, or write the updated numbered list of REMAINING steps.
                            Goal: %s
                            Done so far:
                            %s
                            Old remaining plan: %s""".formatted(state.goal(), done(state),
                            String.join(" | ", state.plan().subList(1, state.plan().size()))));
                    var finalAnswer = FINAL.matcher(reply);
                    if (finalAnswer.find()) {
                        return Map.of(PlanExecuteState.ANSWER, finalAnswer.group(1).strip(), PlanExecuteState.PLAN, List.of());
                    }
                    List<String> plan = TextLists.items(reply, MAX_PLAN_STEPS);
                    return Map.of(PlanExecuteState.PLAN, plan, PlanExecuteState.PLAN_HISTORY, String.join(" | ", plan));
                }))
                .addNode("finish", node_async(state -> Map.of(PlanExecuteState.ANSWER, llm.apply("""
                        Summarise what was achieved towards the goal and what is still open, in two sentences.
                        Goal: %s
                        Done so far:
                        %s""".formatted(state.goal(), done(state))).strip())))
                .addEdge(START, "planner")
                .addConditionalEdges("planner",
                        edge_async(state -> state.plan().isEmpty() ? "finish" : "executor"),
                        Map.of("executor", "executor", "finish", "finish"))
                .addEdge("executor", "replanner")
                .addConditionalEdges("replanner",
                        edge_async(PlanExecuteGraph::afterReplan),
                        Map.of("done", END, "executor", "executor", "finish", "finish"))
                .addEdge("finish", END)
                .compile();
    }

    static String afterReplan(PlanExecuteState state) {
        if (state.answer().isPresent()) {
            return "done";
        }
        return state.plan().isEmpty() || state.steps() >= MAX_STEPS ? "finish" : "executor";
    }

    private static String done(PlanExecuteState state) {
        return state.pastSteps().isEmpty() ? "(nothing yet)" : String.join("\n", state.pastSteps());
    }

    /** Canned replies: the first step reveals a dietary need, and the replanner changes the plan because of it. */
    static ScriptedLlm placeholder() {
        return ScriptedLlm.create()
                .when("You are a planner", """
                        1. Ask the team about dietary needs
                        2. Book a table at Steak Corner for 8
                        3. Send the calendar invite""")
                .when(p -> p.startsWith("You are a replanner") && p.contains("Invite sent"),
                        p -> "FINAL: Lunch for 8 is booked at Green Fork on Friday at 12:30, and the invite is out.")
                .when(p -> p.startsWith("You are a replanner") && p.contains("Booked Green Fork"),
                        p -> "1. Send the calendar invite for Green Fork, Friday 12:30")
                .when(p -> p.startsWith("You are a replanner") && p.contains("vegan"),
                        p -> "1. Book Green Fork (good vegan options) for 8 on Friday at 12:30\n2. Send the calendar invite")
                .when(p -> p.startsWith("You are an executor") && p.contains("Step: Ask the team"),
                        p -> "Two of the eight people are vegan.")
                .when(p -> p.startsWith("You are an executor") && p.contains("Step: Book Green Fork"),
                        p -> "Booked Green Fork for 8 on Friday at 12:30.")
                .when(p -> p.startsWith("You are an executor") && p.contains("Step: Send the calendar invite"),
                        p -> "Invite sent to all 8 people.")
                .otherwise("Partly done; see the steps above.");
    }

    public static void main(String[] args) throws Exception {
        String goal = args.length > 0 ? String.join(" ", args) : "Organise a team lunch for 8 people this Friday.";

        PlanExecuteState result = build(LlmProvider.resolve(placeholder()))
                .invoke(Map.of(PlanExecuteState.GOAL, goal))
                .orElseThrow();

        System.out.println("Goal: " + goal);
        for (int i = 0; i < result.planHistory().size(); i++) {
            System.out.printf("Plan v%d: %s%n", i + 1, result.planHistory().get(i));
        }
        System.out.println("\nExecuted:");
        result.pastSteps().forEach(step -> System.out.println("  " + step));
        System.out.println("\nAnswer after " + result.steps() + " step(s): " + result.answer().orElse("(none)"));
    }
}
