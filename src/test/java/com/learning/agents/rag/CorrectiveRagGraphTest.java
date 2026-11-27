package com.learning.agents.rag;

import com.learning.agents.MainRunner;
import com.learning.agents.llm.HashingEmbeddingModel;
import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CorrectiveRagGraphTest {

    private final VectorRetriever handbook = new VectorRetriever(new HashingEmbeddingModel()).index(EmbeddingRagAgent.HANDBOOK);

    private CorrectiveRagState run(java.util.function.Function<String, String> llm, String question) throws Exception {
        return CorrectiveRagGraph.build(llm, handbook).invoke(Map.of(CorrectiveRagState.QUESTION, question)).orElseThrow();
    }

    @Test
    void rewritesAMissedQueryThenAnswers() throws Exception {
        CorrectiveRagState result = run(CorrectiveRagGraph.placeholder(), "Does my PTO roll forward?");

        assertEquals(1, result.rewrites());
        assertEquals("unused paid leave days carry over next year", result.query());
        assertEquals("leave-policy", result.relevant().getFirst().source());
        assertTrue(result.answer().orElseThrow().contains("[leave-policy]"));
    }

    @Test
    void relevantFirstSearchSkipsTheRewrite() throws Exception {
        CorrectiveRagState result = run(prompt -> prompt.startsWith("Does this document") ? "yes" : "answer", "unused leave carry over");

        assertEquals(0, result.rewrites());
        assertEquals("answer", result.answer().orElseThrow());
    }

    @Test
    void givesUpAfterMaxRewrites() throws Exception {
        var rewrites = new AtomicInteger();
        CorrectiveRagState result = run(prompt -> {
            if (prompt.contains("found nothing useful")) {
                rewrites.incrementAndGet();
                return "leave";
            }
            return "no";
        }, "leave?");

        assertEquals(CorrectiveRagGraph.MAX_REWRITES, rewrites.get());
        assertEquals(CorrectiveRagGraph.NO_ANSWER, result.answer().orElseThrow());
        assertTrue(result.log().getLast().startsWith("give_up"));
    }

    @Test
    void graderNeedsAnExplicitYes() {
        assertTrue(CorrectiveRagGraph.isYes(" **Yes**, it does"));
        assertFalse(CorrectiveRagGraph.isYes("Not really, yes-ish"));
        assertFalse(CorrectiveRagGraph.isYes(""));
    }

    @Test
    void mainShowsBothOutcomes() throws Exception {
        String out = MainRunner.run(CorrectiveRagGraph::main);
        assertTrue(out.contains("rewrite: \"unused paid leave") && out.contains(CorrectiveRagGraph.NO_ANSWER));
        assertTrue(MainRunner.run(CorrectiveRagGraph::main, "laptops").contains("Q: laptops"));
    }
}
