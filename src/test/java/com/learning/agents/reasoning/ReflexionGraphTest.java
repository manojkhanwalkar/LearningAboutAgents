package com.learning.agents.reasoning;

import com.learning.agents.MainRunner;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ReflexionGraphTest {

    @Test
    void lessonsAccumulateUntilTheCheckerPasses() throws Exception {
        ReflexionState result = ReflexionGraph.run(ReflexionGraph.placeholder(), ReflexionGraph::checkPrimes,
                ReflexionGraph.TASK, List.of());

        assertTrue(result.passed());
        assertEquals(3, result.attempts());
        assertEquals(2, result.lessons().size());
        assertEquals("11, 13, 17", result.answer());
        assertTrue(result.history().getFirst().contains("FAIL (15 is not prime)"));
    }

    @Test
    void everyAttemptSeesAllEarlierLessons() throws Exception {
        List<String> attemptPrompts = new ArrayList<>();
        ReflexionGraph.run(prompt -> {
            if (prompt.startsWith("Solve the task")) {
                attemptPrompts.add(prompt);
                return "wrong";
            }
            return "lesson " + attemptPrompts.size();
        }, answer -> Optional.of("nope"), "task", List.of());

        assertEquals(ReflexionGraph.MAX_ATTEMPTS, attemptPrompts.size());
        assertTrue(attemptPrompts.getFirst().contains("(none yet)"));
        assertTrue(attemptPrompts.getLast().contains("- lesson 1\n- lesson 2"));
    }

    @Test
    void priorLessonsCarryIntoANewTask() throws Exception {
        List<String> prompts = new ArrayList<>();
        ReflexionState result = ReflexionGraph.run(prompt -> {
            prompts.add(prompt);
            return "11, 13, 17";
        }, ReflexionGraph::checkPrimes, ReflexionGraph.TASK, List.of("Sort before answering."));

        assertTrue(result.passed());
        assertEquals(1, result.attempts());
        assertTrue(prompts.getFirst().contains("- Sort before answering."));
        assertEquals(List.of("Sort before answering."), result.lessons());
    }

    @Test
    void stopsAtTheAttemptLimit() throws Exception {
        ReflexionState result = ReflexionGraph.run(prompt -> "2, 3, 5", ReflexionGraph::checkPrimes, "t", List.of());

        assertFalse(result.passed());
        assertEquals(ReflexionGraph.MAX_ATTEMPTS, result.attempts());
        assertEquals(ReflexionGraph.MAX_ATTEMPTS - 1, result.lessons().size());
    }

    @Test
    void checkerCoversEachRule() {
        assertEquals(Optional.empty(), ReflexionGraph.checkPrimes("11, 13, 17."));
        assertEquals(Optional.of("expected exactly 3 numbers, got 2"), ReflexionGraph.checkPrimes("11, 13"));
        assertEquals(Optional.of("'x' is not a whole number"), ReflexionGraph.checkPrimes("11, x, 13"));
        assertEquals(Optional.of("7 is not greater than 10"), ReflexionGraph.checkPrimes("7, 11, 13"));
        assertEquals(Optional.of("15 is not prime"), ReflexionGraph.checkPrimes("11, 15, 17"));
        assertEquals(Optional.of("numbers are not in ascending order"), ReflexionGraph.checkPrimes("13, 13, 17"));
        assertFalse(ReflexionGraph.isPrime(1));
        assertTrue(ReflexionGraph.isPrime(2));
        assertFalse(ReflexionGraph.isPrime(49));
    }

    @Test
    void mainShowsLessonsAndTheSolution() throws Exception {
        String out = MainRunner.run(ReflexionGraph::main);
        assertTrue(out.contains("Lessons learned:") && out.contains("Solved in 3 attempt(s): 11, 13, 17"));
        assertTrue(MainRunner.run(ReflexionGraph::main, "other", "task").contains("Task: other task"));
    }
}
