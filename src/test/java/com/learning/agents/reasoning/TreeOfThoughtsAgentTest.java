package com.learning.agents.reasoning;

import com.learning.agents.MainRunner;
import com.learning.agents.reasoning.TreeOfThoughtsAgent.Node;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TreeOfThoughtsAgentTest {

    @Test
    void beamSearchFindsTwentyFour() {
        var result = new TreeOfThoughtsAgent(TreeOfThoughtsAgent.placeholder()).solve(List.of(4, 9, 10, 13));

        // Two paths reach [4 6]; either order of the first two steps is a valid solution.
        var steps = result.solution().orElseThrow();
        assertEquals(3, steps.size());
        assertTrue(steps.containsAll(List.of("10 - 4 = 6", "13 - 9 = 4")));
        assertEquals("4 * 6 = 24", steps.getLast());
        assertEquals(3, result.trace().size());
    }

    /** Steps and trace lines are data (parsed back and asserted), so they must not use the locale's digits. */
    @ParameterizedTest
    @ValueSource(strings = {"ar-EG", "fa-IR"})
    void stepsAndTraceUseAsciiDigitsUnderAnyDefaultLocale(String tag) {
        Locale saved = Locale.getDefault();
        try {
            Locale.setDefault(Locale.forLanguageTag(tag));
            var result = new TreeOfThoughtsAgent(TreeOfThoughtsAgent.placeholder()).solve(List.of(4, 9, 10, 13));

            assertEquals("4 * 6 = 24", result.solution().orElseThrow().getLast());
            assertTrue(result.trace().getFirst().startsWith("depth 1: "), result.trace().getFirst());
        } finally {
            Locale.setDefault(saved);
        }
    }

    @Test
    void stepsWithWrongArithmeticOrUnavailableNumbersAreRejected() {
        List<Integer> left = List.of(4, 9, 10, 13);
        assertEquals(Optional.empty(), TreeOfThoughtsAgent.parseStep("9 * 10 = 99", left));
        assertEquals(Optional.empty(), TreeOfThoughtsAgent.parseStep("4 + 4 = 8", left));
        assertEquals(Optional.empty(), TreeOfThoughtsAgent.parseStep("13 / 4 = 3", left));
        assertEquals(Optional.empty(), TreeOfThoughtsAgent.parseStep("10 / 0 = 0", List.of(10, 0)));
        assertEquals(Optional.empty(), TreeOfThoughtsAgent.parseStep("no step here", left));
        assertEquals("4 + 4 = 8", TreeOfThoughtsAgent.parseStep("4 + 4 = 8", List.of(4, 4)).orElseThrow().toString());
        assertEquals("9 * 4 = 36", TreeOfThoughtsAgent.parseStep("9 × 4 = 36", left).orElseThrow().toString());
        assertEquals("10 / 2 = 5", TreeOfThoughtsAgent.parseStep("10 ÷ 2 = 5", List.of(10, 2)).orElseThrow().toString());
    }

    @Test
    void hugeOrOverflowingNumbersAreRejectedNotCrashing() {
        assertEquals(Optional.empty(), TreeOfThoughtsAgent.parseStep("4 * 9 = 99999999999", List.of(4, 9)));
        assertEquals(Optional.empty(), TreeOfThoughtsAgent.parseStep("65536 * 65536 = 0", List.of(65536, 65536)));
        assertEquals(Optional.empty(), TreeOfThoughtsAgent.parseStep("999999 * 999999 = 999999", List.of(999999, 999999)));
        // A proposal the model got badly wrong must not crash the whole search.
        var result = new TreeOfThoughtsAgent(p -> p.contains("Numbers left: ") ? "4 * 9 = 99999999999999" : "likely")
                .solve(List.of(4, 9, 10, 13));
        assertTrue(result.solution().isEmpty());
    }

    @Test
    void alreadyTwentyFourIsSolvedWithNoSteps() {
        assertEquals(Optional.of(List.of()), new TreeOfThoughtsAgent(p -> "").solve(List.of(24)).solution());
    }

    @Test
    void singleNumbersThatMissAreNeverSentForRating() {
        var ratedSingles = new AtomicInteger();
        new TreeOfThoughtsAgent(prompt -> {
            if (prompt.contains("Numbers left: ")) {
                return "1 + 1 = 2\n2 + 1 = 3\n2 + 2 = 4";
            }
            String numbers = prompt.substring(prompt.indexOf("Numbers: ") + 9).split("\n")[0].strip();
            if (!numbers.contains(" ")) {
                ratedSingles.incrementAndGet();   // a one-number position reached the model
            }
            return "likely";
        }).solve(List.of(1, 1, 2));
        assertEquals(0, ratedSingles.get());
    }

    @Test
    void applyReplacesTheOperandsWithTheResult() {
        var step = TreeOfThoughtsAgent.parseStep("10 - 4 = 6", List.of(4, 9, 10, 13)).orElseThrow();
        Node next = TreeOfThoughtsAgent.apply(new Node(List.of(4, 9, 10, 13), List.of()), step);
        assertEquals("6 9 13", next.key());
        assertEquals(List.of("10 - 4 = 6"), next.steps());
    }

    @Test
    void samePositionReachedTwiceIsRatedOnce() {
        var ratings = new AtomicInteger();
        new TreeOfThoughtsAgent(prompt -> {
            if (prompt.contains("Numbers left: ")) {
                return "1 + 2 = 3\n2 + 1 = 3";
            }
            ratings.incrementAndGet();
            return "likely";
        }).solve(List.of(1, 2, 3, 4));

        // Both proposals lead to [3 3 4]; it is rated once. Depth 2 has no valid proposals, so nothing else is rated.
        assertEquals(1, ratings.get());
    }

    @Test
    void impossiblePositionsArePrunedAndTheSearchCanFail() {
        var result = new TreeOfThoughtsAgent(prompt -> prompt.contains("Numbers left: ") ? "1 + 1 = 2" : "impossible")
                .solve(List.of(1, 1, 1, 1));

        assertTrue(result.solution().isEmpty());
        assertTrue(result.trace().getLast().contains("kept []"));
    }

    @Test
    void ratingWordsMapToScores() {
        var node = new Node(List.of(4, 6), List.of());
        assertEquals(3, new TreeOfThoughtsAgent(p -> "Sure!").rate(node));
        assertEquals(2, new TreeOfThoughtsAgent(p -> "likely").rate(node));
        assertEquals(0, new TreeOfThoughtsAgent(p -> "Impossible, not sure").rate(node));
        assertEquals(0, new TreeOfThoughtsAgent(p -> "maybe").rate(node));
        assertEquals(0, new TreeOfThoughtsAgent(p -> "unlikely").rate(node));
        assertEquals(0, new TreeOfThoughtsAgent(p -> "I'm not sure").rate(node));
        assertEquals(3, new TreeOfThoughtsAgent(p -> "**Sure**").rate(node));
    }

    @Test
    void mainPrintsTheSolution() throws Exception {
        assertTrue(MainRunner.run(TreeOfThoughtsAgent::main).contains("then 4 * 6 = 24"));
        assertTrue(MainRunner.run(TreeOfThoughtsAgent::main, "1", "1", "1", "1").contains("No solution"));
        assertTrue(MainRunner.run(TreeOfThoughtsAgent::main, "3", "3", "8", "8", "1").contains("Usage"));
        assertTrue(MainRunner.run(TreeOfThoughtsAgent::main, "a", "b", "c", "d").contains("Usage"));
    }
}
