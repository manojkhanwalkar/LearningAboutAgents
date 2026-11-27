package com.learning.agents.graph;

import com.learning.agents.MainRunner;
import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PromptChainingGraphTest {

    @Test
    void runsEveryStepInOrderWhenOutlinePassesTheGate() throws Exception {
        PromptChainState result = PromptChainingGraph.build(PromptChainingGraph.placeholder())
                .invoke(Map.of(PromptChainState.TOPIC, "chaining"))
                .orElseThrow();

        assertTrue(result.finalText().orElseThrow().contains("(polished)"));
        assertEquals(3, result.log().size());
        assertEquals(1, result.outlineAttempts());
    }

    @Test
    void retriesBadOutlineThenContinues() throws Exception {
        var outlineCalls = new AtomicInteger();
        PromptChainState result = PromptChainingGraph.build(prompt -> {
                    if (prompt.startsWith("Create an outline")) {
                        return outlineCalls.incrementAndGet() == 1 ? "just one line" : "- a\n- b\n- c";
                    }
                    return prompt.startsWith("Polish") ? "final" : "draft";
                })
                .invoke(Map.of(PromptChainState.TOPIC, "x"))
                .orElseThrow();

        assertEquals("final", result.finalText().orElseThrow());
        assertEquals(2, result.outlineAttempts());
    }

    @Test
    void givesUpWithoutDraftingWhenOutlineNeverPasses() throws Exception {
        var calls = new AtomicInteger();
        PromptChainState result = PromptChainingGraph.build(prompt -> {
                    calls.incrementAndGet();
                    return "no bullets here";
                })
                .invoke(Map.of(PromptChainState.TOPIC, "x"))
                .orElseThrow();

        assertTrue(result.finalText().isEmpty());
        assertTrue(result.draft().isEmpty());
        assertEquals(PromptChainingGraph.MAX_OUTLINE_ATTEMPTS, calls.get());
    }

    @Test
    void countsOnlyDashBullets() {
        assertEquals(2, PromptChainingGraph.countPoints("- one\n  - two\n* three\n4. four"));
    }

    @Test
    void mainPrintsPolishedText() throws Exception {
        assertTrue(MainRunner.run(PromptChainingGraph::main).contains("Why Small Prompts Win"));
    }
}
