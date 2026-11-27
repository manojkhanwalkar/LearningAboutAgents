package com.learning.agents.graph;

import com.learning.agents.MainRunner;
import org.bsc.langgraph4j.RunnableConfig;
import org.junit.jupiter.api.Test;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class HumanApprovalGraphTest {

    @Test
    void pausesBeforeReviewAndDoesNotActOnItsOwn() throws Exception {
        var executeCalls = new AtomicInteger();
        var graph = HumanApprovalGraph.build(prompt -> {
            if (prompt.contains("was approved")) {
                executeCalls.incrementAndGet();
            }
            return "ACTION: delete the logs";
        });
        var config = RunnableConfig.builder().threadId("t1").build();

        ApprovalState paused = graph.invoke(Map.of(ApprovalState.REQUEST, "clean up"), config).orElseThrow();

        assertEquals("ACTION: delete the logs", paused.proposal());
        assertTrue(paused.result().isEmpty());
        assertEquals(HumanApprovalGraph.REVIEW_NODE, graph.getState(config).nextNodeId());
        assertEquals(0, executeCalls.get());
    }

    @Test
    void approvedActionIsExecutedAfterResume() throws Exception {
        ApprovalState result = HumanApprovalGraph.run(HumanApprovalGraph.placeholder(), "fix staging", proposal -> true);

        assertTrue(result.approved());
        assertEquals("Done: web-staging-1 was restarted.", result.result().orElseThrow());
    }

    @Test
    void rejectedActionIsCancelled() throws Exception {
        ApprovalState result = HumanApprovalGraph.run(HumanApprovalGraph.placeholder(), "fix staging", proposal -> false);

        assertFalse(result.approved());
        assertTrue(result.result().orElseThrow().startsWith("Cancelled by reviewer"));
    }

    @Test
    void consoleReviewerOnlyAcceptsExplicitYes() {
        assertTrue(HumanApprovalGraph.consoleReviewer(reader(" YES \n")).test("p"));
        assertFalse(HumanApprovalGraph.consoleReviewer(reader("sure\n")).test("p"));
        assertFalse(HumanApprovalGraph.consoleReviewer(reader("")).test("p"));
        assertFalse(HumanApprovalGraph.consoleReviewer(new BufferedReader(new Reader() {
            @Override
            public int read(char[] buffer, int off, int len) throws IOException {
                throw new IOException("closed");
            }

            @Override
            public void close() {
            }
        })).test("p"));
    }

    @Test
    void mainExecutesOnYesAndCancelsOnEndOfInput() throws Exception {
        assertTrue(MainRunner.runWithInput(HumanApprovalGraph::main, "y\n").contains("Done: web-staging-1"));
        assertTrue(MainRunner.runWithInput(HumanApprovalGraph::main, "").contains("Cancelled by reviewer"));
    }

    private static BufferedReader reader(String text) {
        return new BufferedReader(new StringReader(text));
    }
}
