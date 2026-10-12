package com.learning.agents.graph;

import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Exercises the graph wiring with a fake LLM, so no Ollama server is needed.
 */
class WriterReviewerGraphTest {

    @Test
    void stopsWhenReviewerApproves() throws Exception {
        WriterState result = WriterReviewerGraph.build(prompt -> prompt.contains("strict editor") ? "APPROVED" : "a draft")
                .invoke(Map.of(WriterState.TOPIC, "testing"))
                .orElseThrow();

        assertEquals(1, result.revisions());
        assertEquals("a draft", result.draft().orElseThrow());
        assertEquals(2, result.log().size());
    }

    @Test
    void stopsAtRevisionLimitWhenNeverApproved() throws Exception {
        AtomicInteger writerCalls = new AtomicInteger();
        WriterState result = WriterReviewerGraph.build(prompt -> {
                    if (prompt.contains("strict editor")) {
                        return "Needs more detail.";
                    }
                    return "draft " + writerCalls.incrementAndGet();
                })
                .invoke(Map.of(WriterState.TOPIC, "testing"))
                .orElseThrow();

        assertEquals(WriterReviewerGraph.MAX_REVISIONS, result.revisions());
        assertEquals("draft " + WriterReviewerGraph.MAX_REVISIONS, result.draft().orElseThrow());
    }
}
