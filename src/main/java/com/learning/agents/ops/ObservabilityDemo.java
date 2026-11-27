package com.learning.agents.ops;

import com.learning.agents.graph.RouterGraph;
import com.learning.agents.graph.RouterState;
import com.learning.agents.graph.SupervisorGraph;
import com.learning.agents.graph.SupervisorState;
import com.learning.agents.llm.LlmProvider;
import com.learning.agents.llm.ScriptedLlm;
import org.bsc.langgraph4j.GraphInput;
import org.bsc.langgraph4j.RunnableConfig;

import java.util.Map;
import java.util.function.Function;

/**
 * Runs two existing samples through {@link ObservedLlm}: the router, to show a per-call trace (latency, sizes,
 * estimated tokens) with no change to the router's code, and a supervisor team whose writer keeps producing long
 * drafts, to show a token budget stopping a run before its turn limit would.
 */
public class ObservabilityDemo {

    static final int RUNAWAY_BUDGET = 1_500;
    /** The longest reply the budget assumes; with Ollama, set {@code numPredict} to the same so the budget holds. */
    static final int MAX_REPLY_TOKENS = 300;

    /** Canned router replies (the router's own placeholder is package-private to its sample). */
    static ScriptedLlm routerPlaceholder() {
        return ScriptedLlm.create()
                .when(p -> p.startsWith("Classify") && p.contains("charged"), p -> "billing")
                .when(p -> p.startsWith("Classify"), p -> "technical")
                .when("billing specialist", "I can see a duplicate charge; a refund has been requested.")
                .otherwise("1. Restart the app. 2. Clear the cache. 3. Reinstall if it persists.");
    }

    /** A supervisor that never says FINISH and a writer that writes a lot: the kind of run a budget exists for. */
    static ScriptedLlm runawayPlaceholder() {
        return ScriptedLlm.create()
                .when("You are a supervisor", "NEXT: writer")
                .when("You are a writer", "Here is another, longer draft. " + "It covers every angle in great detail. ".repeat(25));
    }

    static void printTrace(String title, ObservedLlm observed) {
        System.out.println(title);
        System.out.println("  call  latency  prompt  reply  ~tokens");
        for (var r : observed.records()) {
            System.out.printf("  #%-4d %5d ms %6d %6d %8d%s%n", r.seq(), r.latencyMs(), r.promptChars(), r.replyChars(),
                    r.estimatedTokens(), (r.failed() ? "  FAILED" : "") + (r.replyOverran() ? "  REPLY OVER LIMIT" : ""));
        }
        System.out.printf("  total: %d call(s), ~%d tokens of %s budget, %d ms%n%n", observed.calls(), observed.usedTokens(),
                observed.budgetTokens() == Integer.MAX_VALUE ? "no" : String.valueOf(observed.budgetTokens()),
                observed.totalLatencyMs());
    }

    /** How a budgeted run ended, and the observer holding its trace. */
    record BudgetedRun(String outcome, ObservedLlm observed) {
    }

    /** Runs the supervisor under a budget and reports what ended it: the budget, or a normal finish. */
    static BudgetedRun runWithBudget(Function<String, String> llm, int budget) throws Exception {
        var observed = new ObservedLlm(llm, budget, MAX_REPLY_TOKENS);
        return new BudgetedRun(supervise(observed), observed);
    }

    private static String supervise(ObservedLlm observed) throws Exception {
        try {
            SupervisorState result = SupervisorGraph.build(observed)
                    .invoke(GraphInput.args(Map.of(SupervisorState.REQUEST, "Write a product description.")),
                            RunnableConfig.builder().build())
                    .orElseThrow();
            return "finished normally after " + result.turns() + " supervisor turn(s)";
        } catch (RuntimeException e) {
            for (Throwable t = e; t != null; t = t.getCause()) {
                if (t instanceof ObservedLlm.BudgetExceededException budgetHit) {
                    return "stopped by the budget: " + budgetHit.getMessage();
                }
            }
            throw e;
        }
    }

    public static void main(String[] args) throws Exception {
        var router = new ObservedLlm(LlmProvider.resolve(routerPlaceholder()), Integer.MAX_VALUE);
        var graph = RouterGraph.build(router);
        String[] queries = args.length > 0
                ? new String[]{String.join(" ", args)}
                : new String[]{"I was charged twice this month", "The app shows error 500 on login"};
        for (String query : queries) {
            graph.invoke(GraphInput.args(Map.of(RouterState.QUERY, query)), RunnableConfig.builder().build());
        }
        printTrace("Router, observed (no budget):", router);

        BudgetedRun run = runWithBudget(LlmProvider.resolve(runawayPlaceholder()), RUNAWAY_BUDGET);
        printTrace("Runaway supervisor with a " + RUNAWAY_BUDGET + "-token budget:", run.observed());
        System.out.println("Outcome: " + run.outcome());
    }
}
