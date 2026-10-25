package com.learning.agents.graph;

import java.math.BigDecimal;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.Function;
import java.util.regex.Pattern;

/**
 * The tools available to {@link ReActAgentGraph}. In text-based ReAct the model sees only each tool's
 * name and description, and calls it by writing {@code Action: name[input]}.
 */
public final class ReActTools {

    /** A tool the model can invoke by name with a single string input. */
    public record ReActTool(String name, String description, Function<String, String> fn) {
    }

    /** A tiny, fixed "knowledge base" so the example is deterministic and works offline. */
    private static final Map<String, String> FACTS = Map.of(
            "eiffel tower", "The Eiffel Tower is 330 metres tall.",
            "burj khalifa", "The Burj Khalifa is 828 metres tall.",
            "mount everest", "Mount Everest is 8849 metres tall.",
            "moon", "The average distance from the Earth to the Moon is 384400 km.",
            "speed of light", "The speed of light is about 299792 km per second."
    );

    private static final Pattern BINARY_EXPR =
            Pattern.compile("^\\s*(-?\\d+(?:\\.\\d+)?)\\s*([+\\-*/])\\s*(-?\\d+(?:\\.\\d+)?)\\s*$");

    private ReActTools() {
    }

    public static List<ReActTool> defaults() {
        return List.of(
                new ReActTool("lookup", "look up a fact about a landmark or physical quantity, e.g. lookup[Eiffel Tower]",
                        ReActTools::lookup),
                new ReActTool("calculate", "evaluate one arithmetic operation on two numbers (+ - * /), e.g. calculate[828 / 330]",
                        ReActTools::calculate));
    }

    static String lookup(String query) {
        String q = query.toLowerCase(Locale.ROOT).trim();
        return FACTS.entrySet().stream()
                .filter(e -> q.contains(e.getKey()) || e.getKey().contains(q))
                .map(Map.Entry::getValue)
                .findFirst()
                .orElse("No fact found for \"%s\". Known topics: %s".formatted(query, String.join(", ", FACTS.keySet())));
    }

    static String calculate(String expression) {
        var m = BINARY_EXPR.matcher(expression);
        if (!m.matches()) {
            return "Error: expected '<number> <op> <number>', got \"%s\"".formatted(expression);
        }
        double a = Double.parseDouble(m.group(1));
        double b = Double.parseDouble(m.group(3));
        if (!Double.isFinite(a) || !Double.isFinite(b)) {
            return "Error: number is out of range";
        }
        if (m.group(2).equals("/") && b == 0) {
            return "Error: division by zero";
        }
        double result = switch (m.group(2)) {
            case "+" -> a + b;
            case "-" -> a - b;
            case "*" -> a * b;
            default -> a / b;
        };
        if (!Double.isFinite(result)) {
            return "Error: result is out of range";
        }
        if (result == Math.rint(result)) {
            // Exact for any magnitude; a (long) cast would silently clamp values beyond Long.MAX_VALUE.
            return new BigDecimal(result).toPlainString();
        }
        return String.format(Locale.ROOT, "%.4f", result);
    }
}
