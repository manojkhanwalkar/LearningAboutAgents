package com.learning.agents.reasoning;

import com.learning.agents.MainRunner;
import com.learning.agents.llm.TextLists;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PlanExecuteGraphTest {

    private static PlanExecuteState run(java.util.function.Function<String, String> llm) throws Exception {
        return PlanExecuteGraph.build(llm).invoke(Map.of(PlanExecuteState.GOAL, "goal")).orElseThrow();
    }

    @Test
    void replanningChangesThePlanAfterAStepRevealsSomething() throws Exception {
        PlanExecuteState result = run(PlanExecuteGraph.placeholder());

        assertEquals(3, result.steps());
        assertEquals(3, result.planHistory().size());
        assertTrue(result.planHistory().getFirst().contains("Steak Corner"));
        assertTrue(result.planHistory().get(1).contains("Green Fork"));
        assertTrue(result.answer().orElseThrow().contains("Green Fork"));
    }

    @Test
    void stepLimitEndsInASummaryAnswer() throws Exception {
        var executorCalls = new AtomicInteger();
        PlanExecuteState result = run(prompt -> {
            if (prompt.startsWith("You are an executor")) {
                executorCalls.incrementAndGet();
                return "done";
            }
            if (prompt.startsWith("Summarise")) {
                return "summary";
            }
            return "1. another step";   // planner and replanner never finish
        });

        assertEquals(PlanExecuteGraph.MAX_STEPS, executorCalls.get());
        assertEquals("summary", result.answer().orElseThrow());
    }

    @Test
    void emptyPlanGoesStraightToFinish() throws Exception {
        var executorCalls = new AtomicInteger();
        PlanExecuteState result = run(prompt -> {
            if (prompt.startsWith("You are an executor")) {
                executorCalls.incrementAndGet();
            }
            return prompt.startsWith("Summarise") ? "nothing to do" : "Here is the plan:";
        });

        assertEquals(0, executorCalls.get());
        assertEquals("nothing to do", result.answer().orElseThrow());
    }

    @Test
    void replanThatEmptiesThePlanWithoutFinalAlsoFinishes() throws Exception {
        PlanExecuteState result = run(prompt -> {
            if (prompt.startsWith("You are a planner")) {
                return "1. only step";
            }
            if (prompt.startsWith("Summarise")) {
                return "wrapped up";
            }
            return prompt.startsWith("You are a replanner") ? "" : "ok";
        });

        assertEquals(1, result.steps());
        assertEquals("wrapped up", result.answer().orElseThrow());
    }

    @Test
    void finalIsCaseInsensitiveAndPlansAreCapped() throws Exception {
        PlanExecuteState result = run(prompt -> {
            if (prompt.startsWith("You are a planner")) {
                return "1. a\n2. b\n3. c\n4. d\n5. e\n6. f\n7. g";
            }
            return prompt.startsWith("You are a replanner") ? "final: all good" : "ok";
        });

        assertEquals(PlanExecuteGraph.MAX_PLAN_STEPS, result.planHistory().getFirst().split(" \\| ").length);
        assertEquals("all good", result.answer().orElseThrow());
    }

    @Test
    void textListsStripMarkersAndPreambles() {
        assertEquals(List.of("a", "b", "c"), TextLists.items("Plan:\n1) a\n- b\n\n• c\n* d", 3));
    }

    @Test
    void mainShowsThePlanChanging() throws Exception {
        String out = MainRunner.run(PlanExecuteGraph::main);
        assertTrue(out.contains("Plan v2: Book Green Fork") && out.contains("Answer after 3 step(s)"));
    }
}
