package com.learning.agents.graph;

import com.learning.agents.MainRunner;
import org.bsc.langgraph4j.GraphInput;
import org.bsc.langgraph4j.RunnableConfig;
import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PromptChainingGraphTest {

    @Test
    void runsEveryStepInOrderWhenOutlinePassesTheGate() throws Exception {
        PromptChainState result = PromptChainingGraph.build(PromptChainingGraph.placeholder())
                .invoke(GraphInput.args(Map.of(PromptChainState.TOPIC, "chaining")), RunnableConfig.builder().build())
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
                .invoke(GraphInput.args(Map.of(PromptChainState.TOPIC, "x")), RunnableConfig.builder().build())
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
                .invoke(GraphInput.args(Map.of(PromptChainState.TOPIC, "x")), RunnableConfig.builder().build())
                .orElseThrow();

        assertTrue(result.finalText().isEmpty());
        assertTrue(result.draft().isEmpty());
        assertEquals(PromptChainingGraph.MAX_OUTLINE_ATTEMPTS, calls.get());
    }

    @Test
    void countsTheBulletStylesModelsUse() {
        assertEquals(4, PromptChainingGraph.countPoints("- one\n  * two\n• three\n4. four\nplain line\n-nospace"));
    }

    @Test
    void retryPromptExplainsWhatWasWrong() throws Exception {
        var prompts = new java.util.ArrayList<String>();
        PromptChainingGraph.build(prompt -> {
                    prompts.add(prompt);
                    return "- only one";
                })
                .invoke(GraphInput.args(Map.of(PromptChainState.TOPIC, "x")), RunnableConfig.builder().build())
                .orElseThrow();

        assertFalse(prompts.getFirst().contains("previous outline"));
        assertTrue(prompts.get(1).contains("Your previous outline had only 1 point(s)"));
    }

    @Test
    void mainPrintsPolishedText() throws Exception {
        assertTrue(MainRunner.run(PromptChainingGraph::main).contains("Why Small Prompts Win"));
    }
}
