package com.learning.agents.reasoning;

import com.learning.agents.llm.LlmProvider;
import com.learning.agents.llm.ScriptedLlm;
import org.bsc.langgraph4j.CompiledGraph;
import org.bsc.langgraph4j.GraphInput;
import org.bsc.langgraph4j.GraphStateException;
import org.bsc.langgraph4j.RunnableConfig;
import org.bsc.langgraph4j.StateGraph;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.function.Function;
import java.util.stream.Collectors;

import static org.bsc.langgraph4j.StateGraph.END;
import static org.bsc.langgraph4j.StateGraph.START;
import static org.bsc.langgraph4j.action.AsyncEdgeAction.edge_async;
import static org.bsc.langgraph4j.action.AsyncNodeAction.node_async;

/**
 * Reflexion (Shinn et al., 2023): learn from failure in words. Each attempt is checked by <em>code</em>; after a
 * failure the model writes a short lesson ("15 is not prime, check divisibility"), and every later attempt sees all
 * lessons so far. Compare with {@code graph.WriterReviewerGraph}, where a model critiques one draft: here the judge is
 * an objective checker, and what carries forward is a growing list of lessons rather than one piece of feedback.
 * Lessons can also be kept and passed to the next task (see {@link #run}), which is the "memory" half of Reflexion.
 *
 * <pre>
 *   START → attempt → check ─┬─ passed / attempt limit → END
 *              ↑             └─ failed → reflect ─┐
 *              └──────────────────────────────────┘
 * </pre>
 */
public class ReflexionGraph {

    static final int MAX_ATTEMPTS = 3;
    static final String TASK = "List three prime numbers greater than 10, comma-separated, in ascending order.";

    /**
     * Builds the graph around a pluggable LLM and checker so it can be tested without Ollama.
     *
     * @param llm     maps a prompt to a model reply
     * @param checker returns an error message for a wrong answer, or empty if it is correct
     */
    public static CompiledGraph<ReflexionState> build(Function<String, String> llm,
                                                      Function<String, Optional<String>> checker)
            throws GraphStateException {
        return new StateGraph<>(ReflexionState.SCHEMA, ReflexionState::new)
                .addNode("attempt", node_async(state -> {
                    String lessons = state.lessons().isEmpty() ? "(none yet)"
                            : state.lessons().stream().map(l -> "- " + l).collect(Collectors.joining("\n"));
                    String answer = llm.apply("""
                            Solve the task. Reply with the answer only.
                            Lessons from your earlier failed attempts (apply all of them):
                            %s
                            Task: %s""".formatted(lessons, state.task())).strip();
                    return Map.of(ReflexionState.ANSWER, answer, ReflexionState.ATTEMPTS, state.attempts() + 1);
                }))
                .addNode("check", node_async(state -> {
                    Optional<String> error = checker.apply(state.answer());
                    return Map.of(ReflexionState.ERROR, error.orElse(""),
                            ReflexionState.HISTORY, String.format(Locale.ROOT, "attempt %d: %s → %s",
                                    state.attempts(), state.answer(), error.map(e -> "FAIL (" + e + ")").orElse("PASS")));
                }))
                .addNode("reflect", node_async(state -> Map.of(ReflexionState.LESSONS, llm.apply("""
                        Your answer to the task was wrong.
                        Task: %s
                        Your answer: %s
                        Checker says: %s
                        Write ONE short, general lesson that would prevent this mistake next time.""".formatted(
                        state.task(), state.answer(), state.error().orElse(""))).strip())))
                .addEdge(START, "attempt")
                .addEdge("attempt", "check")
                .addConditionalEdges("check",
                        edge_async(ReflexionGraph::afterCheck),
                        Map.of("done", END, "reflect", "reflect"))
                .addEdge("reflect", "attempt")
                .compile();
    }

    /** Routes after the checker: done when the attempt passed or the attempt limit is reached, else reflect. */
    static String afterCheck(ReflexionState state) {
        return state.passed() || state.attempts() >= MAX_ATTEMPTS ? "done" : "reflect";
    }

    /**
     * Runs one task, starting from lessons learned on earlier tasks.
     *
     * @return the final state; its {@link ReflexionState#lessons()} include the prior ones, ready for the next task
     */
    static ReflexionState run(Function<String, String> llm, Function<String, Optional<String>> checker,
                              String task, List<String> priorLessons) throws Exception {
        return build(llm, checker)
                .invoke(GraphInput.args(Map.of(ReflexionState.TASK, task, ReflexionState.LESSONS, new ArrayList<>(priorLessons))), RunnableConfig.builder().build())
                .orElseThrow();
    }

    /**
     * The checker for {@link #TASK}, in plain code: exactly three comma-separated integers, each prime, each above
     * 10, strictly ascending.
     */
    static Optional<String> checkPrimes(String answer) {
        String[] parts = answer.strip().replaceFirst("\\.$", "").split("\\s*,\\s*");
        if (parts.length != 3) {
            return Optional.of("expected exactly 3 numbers, got " + parts.length);
        }
        List<Integer> numbers = new ArrayList<>();
        for (String part : parts) {
            try {
                numbers.add(Integer.parseInt(part));
            } catch (NumberFormatException _) {
                return Optional.of("'" + part + "' is not a whole number");
            }
        }
        for (int n : numbers) {
            if (n <= 10) {
                return Optional.of(n + " is not greater than 10");
            }
            if (!isPrime(n)) {
                return Optional.of(n + " is not prime");
            }
        }
        for (int i = 1; i < numbers.size(); i++) {
            if (numbers.get(i) <= numbers.get(i - 1)) {
                return Optional.of("numbers are not in ascending order");
            }
        }
        return Optional.empty();
    }

    /** Trial division up to the square root; numbers below 2 are not prime. */
    static boolean isPrime(int n) {
        if (n < 2) {
            return false;
        }
        for (int d = 2; (long) d * d <= n; d++) {
            if (n % d == 0) {
                return false;
            }
        }
        return true;
    }

    /** Canned replies: the first attempt includes 15, the second is out of order, the third is right. */
    static ScriptedLlm placeholder() {
        return ScriptedLlm.create()
                .when(p -> p.startsWith("Your answer to the task was wrong") && p.contains("not prime"),
                        p -> "Check every number for divisors; odd numbers like 15 can still be composite.")
                .when(p -> p.startsWith("Your answer to the task was wrong") && p.contains("ascending"),
                        p -> "Sort the numbers from smallest to largest before answering.")
                .when(p -> p.startsWith("Your answer to the task was wrong"), p -> "Re-read the task requirements.")
                .when(p -> p.contains("Sort the numbers"), p -> "11, 13, 17")
                .when(p -> p.contains("Check every number"), p -> "13, 11, 17")
                .otherwise("11, 15, 13");
    }

    /** Runs the reflexion loop on its built-in task and prints the attempts, lessons learned and outcome; {@code args} are ignored. */
    public static void main(String[] args) throws Exception {
        // The judge must fit the task: checkPrimes only grades TASK, so this sample doesn't take a task from args.
        if (args.length > 0) {
            System.out.println("Note: this sample's checker only grades its built-in task, so the arguments are ignored.");
        }
        String task = TASK;

        ReflexionState result = run(LlmProvider.resolve(placeholder()), ReflexionGraph::checkPrimes, task, List.of());

        System.out.println("Task: " + task);
        result.history().forEach(line -> System.out.println("  " + line));
        System.out.println("Lessons learned:");
        result.lessons().forEach(lesson -> System.out.println("  - " + lesson));
        System.out.println(result.passed()
                ? "Solved in " + result.attempts() + " attempt(s): " + result.answer()
                : "Not solved within " + MAX_ATTEMPTS + " attempts.");
    }
}
