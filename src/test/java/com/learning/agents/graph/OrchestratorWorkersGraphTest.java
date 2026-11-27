package com.learning.agents.graph;

import com.learning.agents.MainRunner;
import org.bsc.langgraph4j.GraphInput;
import org.bsc.langgraph4j.RunnableConfig;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class OrchestratorWorkersGraphTest {

    @Test
    void runsOneWorkerPerSubtaskThenSynthesizes() throws Exception {
        OrchestratorState result = OrchestratorWorkersGraph.build(OrchestratorWorkersGraph.placeholder())
                .invoke(GraphInput.args(Map.of(OrchestratorState.TASK, "local vs hosted")), RunnableConfig.builder().build())
                .orElseThrow();

        assertEquals(3, result.subtasks().size());
        assertEquals(3, result.results().size());
        assertTrue(result.results().get(2).contains("network"));
        assertTrue(result.report().orElseThrow().contains("privacy"));
    }

    @Test
    void capsTheNumberOfWorkers() throws Exception {
        var workerCalls = new AtomicInteger();
        OrchestratorState result = OrchestratorWorkersGraph.build(prompt -> {
                    if (prompt.startsWith("You are an orchestrator")) {
                        return "a\nb\nc\nd\ne\nf\ng";
                    }
                    if (prompt.startsWith("You are a worker")) {
                        workerCalls.incrementAndGet();
                    }
                    return "ok";
                })
                .invoke(GraphInput.args(Map.of(OrchestratorState.TASK, "big task")), RunnableConfig.builder().build())
                .orElseThrow();

        assertEquals(OrchestratorWorkersGraph.MAX_SUBTASKS, workerCalls.get());
        assertEquals("ok", result.report().orElseThrow());
    }

    @Test
    void identicalWorkerOutputsAreAllKept() throws Exception {
        OrchestratorState result = OrchestratorWorkersGraph.build(prompt -> prompt.startsWith("You are an orchestrator") ? "a\nb\nc" : "same")
                .invoke(GraphInput.args(Map.of(OrchestratorState.TASK, "t")), RunnableConfig.builder().build())
                .orElseThrow();

        assertEquals(List.of("same", "same", "same"), result.results());
    }

    @Test
    void emptyPlanGoesStraightToSynthesis() throws Exception {
        OrchestratorState result = OrchestratorWorkersGraph.build(prompt -> prompt.startsWith("You are an orchestrator") ? "\n\n" : "nothing to do")
                .invoke(GraphInput.args(Map.of(OrchestratorState.TASK, "t")), RunnableConfig.builder().build())
                .orElseThrow();

        assertTrue(result.results().isEmpty());
        assertEquals("nothing to do", result.report().orElseThrow());
    }

    @Test
    void parsePlanStripsListMarkers() {
        assertEquals(List.of("one", "two", "three", "four"),
                OrchestratorWorkersGraph.parsePlan("1. one\n- two\n\n* three\n2) four\nfive"));
        assertEquals(List.of("a", "b"), OrchestratorWorkersGraph.parsePlan("Here are the subtasks:\n1. a\n2. b"));
    }

    @Test
    void mainPrintsPlanAndReport() throws Exception {
        String out = MainRunner.run(OrchestratorWorkersGraph::main);
        assertTrue(out.contains("1. List the costs") && out.contains("Report: "));
    }
}
