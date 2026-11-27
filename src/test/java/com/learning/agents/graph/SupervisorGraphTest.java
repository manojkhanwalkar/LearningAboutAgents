package com.learning.agents.graph;

import com.learning.agents.MainRunner;
import org.bsc.langgraph4j.GraphInput;
import org.bsc.langgraph4j.RunnableConfig;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SupervisorGraphTest {

    @Test
    void supervisorDelegatesUntilFinish() throws Exception {
        SupervisorState result = SupervisorGraph.build(SupervisorGraph.placeholder())
                .invoke(GraphInput.args(Map.of(SupervisorState.REQUEST, "Explain Gemma")), RunnableConfig.builder().build())
                .orElseThrow();

        assertEquals(List.of("researcher", "writer", "critic"),
                result.work().stream().map(entry -> entry.substring(0, entry.indexOf(':'))).toList());
        assertEquals(4, result.turns());
        assertTrue(SupervisorGraph.latestDraft(result.work()).orElseThrow().contains("Ollama"));
    }

    @Test
    void stopsAtTurnLimitWhenSupervisorNeverFinishes() throws Exception {
        SupervisorState result = SupervisorGraph.build(prompt -> prompt.startsWith("You are a supervisor") ? "NEXT: writer" : "draft")
                .invoke(GraphInput.args(Map.of(SupervisorState.REQUEST, "r")), RunnableConfig.builder().build())
                .orElseThrow();

        assertEquals(SupervisorGraph.MAX_TURNS, result.turns());
        assertEquals(SupervisorGraph.MAX_TURNS - 1, result.work().size());
    }

    @Test
    void unknownAgentEndsTheRun() throws Exception {
        SupervisorState result = SupervisorGraph.build(prompt -> "NEXT: hacker")
                .invoke(GraphInput.args(Map.of(SupervisorState.REQUEST, "r")), RunnableConfig.builder().build())
                .orElseThrow();

        assertTrue(result.work().isEmpty());
        assertEquals(Optional.empty(), SupervisorGraph.latestDraft(result.work()));
    }

    @Test
    void parseNextIsCaseInsensitive() {
        assertEquals(Optional.of("critic"), SupervisorGraph.parseNext("I think next: Critic"));
        assertEquals(Optional.empty(), SupervisorGraph.parseNext("FINISH"));
        assertEquals(Optional.of("writer"), SupervisorGraph.parseNext("NEXT: the writer"));
        assertEquals(Optional.of("writer"), SupervisorGraph.parseNext("**NEXT:** writer"));
        assertEquals(Optional.empty(), SupervisorGraph.parseNext("NEXT: nobody"));
    }

    @Test
    void latestDraftPicksTheMostRecentWriterEntry() {
        assertEquals(Optional.of("v2"), SupervisorGraph.latestDraft(List.of("writer: v1", "critic: meh", "writer: v2")));
    }

    @Test
    void mainPrintsTheAnswer() throws Exception {
        assertTrue(MainRunner.run(SupervisorGraph::main).contains("Answer: You can run open models"));
    }
}
