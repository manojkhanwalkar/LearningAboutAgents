package com.learning.agents.reasoning;

import com.learning.agents.MainRunner;
import com.learning.agents.reasoning.TreeOfThoughtsAgent.Node;
import org.junit.jupiter.api.Test;

import java.util.List;
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

        // depth 1: "3 3 4" twice → rated once; depth 2 and 3 add more, but never a duplicate rating per depth.
        assertTrue(ratings.get() <= 3, "rated " + ratings.get() + " times");
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
    }

    @Test
    void mainPrintsTheSolution() throws Exception {
        assertTrue(MainRunner.run(TreeOfThoughtsAgent::main).contains("then 4 * 6 = 24"));
        assertTrue(MainRunner.run(TreeOfThoughtsAgent::main, "1", "1", "1", "1").contains("No solution"));
    }
}
