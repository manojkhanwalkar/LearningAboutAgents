package com.learning.agents.graph;

import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;
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

    /** Longest operand accepted, so results stay short enough to hand back to the model. */
    static final int MAX_DIGITS = 30;

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
        if (digits(m.group(1)) > MAX_DIGITS || digits(m.group(3)) > MAX_DIGITS) {
            return "Error: number is out of range (at most %d digits)".formatted(MAX_DIGITS);
        }
        // Decimal arithmetic, so results are exact (or correctly rounded for division) at any magnitude
        // the digit limit allows, with no silent overflow, underflow or binary rounding.
        BigDecimal a = new BigDecimal(m.group(1));
        BigDecimal b = new BigDecimal(m.group(3));
        if (m.group(2).equals("/") && b.signum() == 0) {
            return "Error: division by zero";
        }
        BigDecimal result = switch (m.group(2)) {
            case "+" -> a.add(b);
            case "-" -> a.subtract(b);
            case "*" -> a.multiply(b);
            default -> a.divide(b, MathContext.DECIMAL128);
        };
        BigDecimal whole = result.stripTrailingZeros();
        return whole.scale() <= 0
                ? whole.toPlainString()
                : result.setScale(4, RoundingMode.HALF_UP).toPlainString();
    }

    private static int digits(String number) {
        return (int) number.chars().filter(Character::isDigit).count();
    }
}
