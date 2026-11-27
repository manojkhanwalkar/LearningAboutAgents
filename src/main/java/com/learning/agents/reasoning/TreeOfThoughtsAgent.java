package com.learning.agents.reasoning;

import com.learning.agents.llm.LlmProvider;
import com.learning.agents.llm.ScriptedLlm;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.function.Function;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Tree of Thoughts (Yao et al., 2023) on the Game of 24: use four numbers and + - * / to make 24. Instead of one chain
 * of reasoning, the agent explores a <em>tree</em>. At each depth the LLM <b>proposes</b> next steps for every kept
 * position; <b>code verifies</b> each step's arithmetic (a step the model miscalculated is thrown away); the LLM
 * <b>rates</b> each new position as sure, likely or impossible; and a beam search keeps the best {@link #BEAM_WIDTH}.
 * After three steps one number is left, and code checks whether it is 24.
 * <p>
 * The split of work is the lesson: the model is good at suggesting and judging, code is good at exact checking and
 * bookkeeping. Each depth costs one proposal call per kept position plus one rating call per new position, so the
 * beam width and {@link #MAX_PROPOSALS} cap the cost.
 *
 * <pre>
 *   [4 9 10 13] ──propose──▶ verify ──rate──▶ keep best 2 ──▶ … (3 levels) ──▶ one number left = 24?
 * </pre>
 */
public class TreeOfThoughtsAgent {

    static final int TARGET = 24;
    static final int BEAM_WIDTH = 2;
    static final int MAX_PROPOSALS = 5;

    private static final Pattern STEP = Pattern.compile("(-?\\d+)\\s*([-+*/×x÷])\\s*(-?\\d+)\\s*=\\s*(-?\\d+)");

    /** A position in the search: the numbers still to use, and the steps taken to get here. */
    record Node(List<Integer> left, List<String> steps) {
        String key() {
            return left.stream().sorted().map(String::valueOf).collect(Collectors.joining(" "));
        }
    }

    /** A verified step: which two numbers were combined, how, and the exact result. */
    record Step(int a, char op, int b, int result) {
        @Override
        public String toString() {
            return "%d %c %d = %d".formatted(a, op, b, result);
        }
    }

    /** The outcome: the steps that reach 24 (if found), and a log of every level of the search. */
    record Result(Optional<List<String>> solution, List<String> trace) {
    }

    private final Function<String, String> llm;

    TreeOfThoughtsAgent(Function<String, String> llm) {
        this.llm = llm;
    }

    /** Runs the beam search from {@code numbers}; stops early when a position reaches exactly 24. */
    public Result solve(List<Integer> numbers) {
        List<String> trace = new ArrayList<>();
        List<Node> beam = List.of(new Node(List.copyOf(numbers), List.of()));
        for (int depth = 1; depth < numbers.size(); depth++) {
            List<Node> children = new ArrayList<>();
            for (Node node : beam) {
                for (Step step : propose(node)) {
                    children.add(apply(node, step));
                }
            }
            for (Node child : children) {
                if (child.left().size() == 1 && child.left().getFirst() == TARGET) {
                    trace.add("depth %d: solved with %s".formatted(depth, child.steps()));
                    return new Result(Optional.of(child.steps()), trace);
                }
            }
            // The same numbers reached by different paths are one position: rate it once and keep one copy.
            Map<String, Node> unique = new LinkedHashMap<>();
            children.forEach(child -> unique.putIfAbsent(child.key(), child));
            Map<Node, Integer> scores = new LinkedHashMap<>();
            unique.values().forEach(child -> scores.put(child, rate(child)));
            beam = scores.entrySet().stream()
                    .filter(e -> e.getValue() > 0)
                    .sorted(Map.Entry.<Node, Integer>comparingByValue().reversed()
                            .thenComparing(e -> e.getKey().key()))
                    .limit(BEAM_WIDTH)
                    .map(Map.Entry::getKey)
                    .toList();
            trace.add("depth %d: %d verified step(s), kept %s".formatted(depth, children.size(),
                    beam.stream().map(n -> "[" + n.key() + "] (" + scores.get(n) + ")").toList()));
            if (beam.isEmpty()) {
                break;
            }
        }
        return new Result(Optional.empty(), trace);
    }

    /** Asks for next steps and keeps only those whose arithmetic is right and whose numbers are available. */
    List<Step> propose(Node node) {
        String reply = llm.apply("""
                Game of 24. Numbers left: %s
                Propose up to %d possible next steps, one per line, each combining two of the numbers, like "4 + 9 = 13".
                Only use + - * / and whole-number results.""".formatted(node.key(), MAX_PROPOSALS));
        return reply.lines()
                .map(line -> parseStep(line, node.left()))
                .flatMap(Optional::stream)
                .limit(MAX_PROPOSALS)
                .toList();
    }

    /** Asks the model how promising a position is; "sure" = 3, "likely" = 2, anything else = 0 (pruned). */
    int rate(Node node) {
        String reply = llm.apply("""
                Game of 24. Can these numbers reach exactly 24 using each once with + - * /?
                Numbers: %s
                Reply with one word: sure, likely or impossible.""".formatted(node.key())).toLowerCase(Locale.ROOT);
        if (reply.contains("impossible")) {
            return 0;
        }
        return reply.contains("sure") ? 3 : reply.contains("likely") ? 2 : 0;
    }

    /**
     * Parses "a op b = c" and checks it: both operands must be among {@code left} (respecting duplicates), the result
     * must be what the model claims, and division must be exact. Returns empty for anything else.
     */
    static Optional<Step> parseStep(String line, List<Integer> left) {
        var m = STEP.matcher(line);
        if (!m.find()) {
            return Optional.empty();
        }
        int a = Integer.parseInt(m.group(1));
        int b = Integer.parseInt(m.group(3));
        int claimed = Integer.parseInt(m.group(4));
        char op = switch (m.group(2)) {
            case "×", "x" -> '*';
            case "÷" -> '/';
            default -> m.group(2).charAt(0);
        };
        List<Integer> pool = new ArrayList<>(left);
        if (!pool.remove(Integer.valueOf(a)) || !pool.remove(Integer.valueOf(b))) {
            return Optional.empty();
        }
        Optional<Integer> actual = switch (op) {
            case '+' -> Optional.of(a + b);
            case '-' -> Optional.of(a - b);
            case '*' -> Optional.of(a * b);
            default -> b != 0 && a % b == 0 ? Optional.of(a / b) : Optional.empty();
        };
        return actual.filter(r -> r == claimed).map(r -> new Step(a, op, b, r));
    }

    /** The position after {@code step}: its two operands replaced by the result. */
    static Node apply(Node node, Step step) {
        List<Integer> left = new ArrayList<>(node.left());
        left.remove(Integer.valueOf(step.a()));
        left.remove(Integer.valueOf(step.b()));
        left.add(step.result());
        List<String> steps = new ArrayList<>(node.steps());
        steps.add(step.toString());
        return new Node(List.copyOf(left), List.copyOf(steps));
    }

    /** Canned proposals and ratings for 4 9 10 13; one proposal has wrong arithmetic so you can see it rejected. */
    static ScriptedLlm placeholder() {
        Map<String, String> proposals = Map.of(
                "4 9 10 13", "10 - 4 = 6\n13 - 9 = 4\n9 * 10 = 99\n4 + 9 = 13",
                "6 9 13", "13 - 9 = 4\n6 + 9 = 15",
                "4 4 10", "10 - 4 = 6\n4 * 4 = 16",
                "4 6", "6 - 4 = 2\n4 * 6 = 24");
        Map<String, String> ratings = Map.of(
                "6 9 13", "likely", "4 4 10", "likely", "10 13 13", "impossible",
                "4 6", "sure", "6 15", "impossible", "6 4", "sure", "10 16", "impossible");
        return ScriptedLlm.create()
                .when(p -> p.contains("Numbers left: "), p -> proposals.getOrDefault(after(p, "Numbers left: "), ""))
                .when(p -> p.contains("Numbers: "), p -> ratings.getOrDefault(after(p, "Numbers: "), "impossible"));
    }

    private static String after(String prompt, String marker) {
        int i = prompt.indexOf(marker) + marker.length();
        return prompt.substring(i, prompt.indexOf('\n', i)).strip();
    }

    public static void main(String[] args) {
        List<Integer> numbers = args.length == 4
                ? List.of(args).stream().map(Integer::parseInt).toList()
                : List.of(4, 9, 10, 13);

        Result result = new TreeOfThoughtsAgent(LlmProvider.resolve(placeholder())).solve(numbers);

        System.out.println("Numbers: " + numbers);
        result.trace().forEach(line -> System.out.println("  " + line));
        System.out.println(result.solution()
                .map(steps -> "Solution: " + String.join(", then ", steps))
                .orElse("No solution found within the beam."));
    }
}
