package com.learning.agents.graph;

import org.bsc.langgraph4j.GraphInput;
import org.bsc.langgraph4j.RunnableConfig;
import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Exercises the ReAct loop with a scripted fake LLM, so no Ollama server is needed.
 */
class ReActAgentGraphTest {

    @Test
    void thoughtActionObservationLoopReachesFinalAnswer() throws Exception {
        // The fake model decides its next step by looking at which Observations are already in the prompt.
        ReActState result = ReActAgentGraph.build(prompt -> {
                    if (!prompt.contains("Observation:")) {
                        return "Thought: I need the Burj Khalifa's height.\nAction: lookup[Burj Khalifa]";
                    }
                    if (!prompt.contains("330 metres")) {
                        return "Thought: Now the Eiffel Tower.\nAction: lookup[Eiffel Tower]";
                    }
                    if (!prompt.contains("Observation: 2.5091")) {
                        return "Thought: Divide the heights.\nAction: calculate[828 / 330]";
                    }
                    return "Thought: I have the ratio.\nFinal Answer: about 2.5 times taller";
                })
                .invoke(GraphInput.args(Map.of(ReActState.QUESTION, "How many times taller is the Burj Khalifa than the Eiffel Tower?")), RunnableConfig.builder().build())
                .orElseThrow();

        assertEquals("about 2.5 times taller", result.answer().orElseThrow());
        assertEquals(4, result.steps());
        // 4 reasoning entries interleaved with 3 observations.
        assertEquals(7, result.transcript().size());
        assertEquals("Observation: 2.5091", result.transcript().get(5));
    }

    @Test
    void stopsAtStepLimitWhenModelNeverAnswers() throws Exception {
        AtomicInteger calls = new AtomicInteger();
        ReActState result = ReActAgentGraph.build(prompt -> {
                    calls.incrementAndGet();
                    return "Thought: hmm.\nAction: lookup[Atlantis]";
                })
                .invoke(GraphInput.args(Map.of(ReActState.QUESTION, "Where is Atlantis?")), RunnableConfig.builder().build())
                .orElseThrow();

        assertTrue(result.answer().isEmpty());
        assertEquals(ReActAgentGraph.MAX_STEPS, calls.get());
    }

    @Test
    void malformedReplyProducesCorrectiveObservation() throws Exception {
        ReActState result = ReActAgentGraph.build(prompt -> prompt.contains("Invalid action")
                        ? "Final Answer: 42"
                        : "I think the answer is 42.")
                .invoke(GraphInput.args(Map.of(ReActState.QUESTION, "What is the answer?")), RunnableConfig.builder().build())
                .orElseThrow();

        assertEquals("42", result.answer().orElseThrow());
        assertTrue(result.transcript().get(1).startsWith("Observation: Invalid action"));
    }

    @Test
    void transcriptKeepsARepeatedStepAndObservation() throws Exception {
        // A model that repeats itself must see (and leave behind) every step; a de-duplicating channel would drop the
        // second identical Action and Observation, so the transcript would no longer match what really happened.
        AtomicInteger calls = new AtomicInteger();
        ReActState result = ReActAgentGraph.build(prompt -> calls.incrementAndGet() <= 2
                        ? "Thought: check the Moon.\nAction: lookup[Moon]"
                        : "Thought: done.\nFinal Answer: 384400 km")
                .invoke(GraphInput.args(Map.of(ReActState.QUESTION, "How far is the Moon?")), RunnableConfig.builder().build())
                .orElseThrow();

        assertEquals("384400 km", result.answer().orElseThrow());
        assertEquals(5, result.transcript().size());
        assertEquals(result.transcript().get(0), result.transcript().get(2));
        assertEquals(result.transcript().get(1), result.transcript().get(3));
    }

    @Test
    void parseExtractsThoughtAndAction() {
        assertEquals(new ReActAgentGraph.Step.Action("need data", "lookup", "Mount Everest"),
                ReActAgentGraph.parse("Thought: need data\nAction: lookup[ Mount Everest ]"));
    }

    @Test
    void parseReadsAFinalAnswerAndTreatsAnUnreadableReplyAsABlankAction() {
        assertEquals(new ReActAgentGraph.Step.FinalAnswer("done", "42"),
                ReActAgentGraph.parse("Thought: done\nFinal Answer: 42"));
        assertEquals(new ReActAgentGraph.Step.Action("", "", ""), ReActAgentGraph.parse("I am not sure."));
    }

    @Test
    void parseIsLinearOnLongWhitespaceRuns() {
        // The old Thought:\s*(.*?)\s*(?=...) pattern rescanned the whitespace run once per character: quadratic.
        String reply = "Thought: a" + " ".repeat(200_000) + "b";
        assertTimeoutPreemptively(Duration.ofSeconds(1), () -> assertEquals("a" + " ".repeat(200_000) + "b",
                ReActAgentGraph.parse(reply).thought()));
    }

    @Test
    void truncatesHallucinatedObservations() {
        String reply = "Thought: x\nAction: lookup[Moon]\nObservation: made up\nFinal Answer: wrong";
        assertEquals("Thought: x\nAction: lookup[Moon]", ReActAgentGraph.truncateAtObservation(reply));
    }

    @Test
    void calculateHandlesOperatorsAndErrors() {
        assertEquals("12", ReActTools.calculate("3 * 4"));
        assertEquals("2.5091", ReActTools.calculate("828 / 330"));
        assertEquals("Error: division by zero", ReActTools.calculate("1 / 0"));
        assertTrue(ReActTools.calculate("sqrt(2)").startsWith("Error"));
    }

    @Test
    void calculateRejectsOutOfRangeInsteadOfMisreporting() {
        String huge = "9".repeat(400);
        assertTrue(ReActTools.calculate(huge + " + 1").startsWith("Error: number is out of range"));
        String tiny = "0." + "0".repeat(400) + "1";
        assertTrue(ReActTools.calculate("5 / " + tiny).startsWith("Error: number is out of range"));
    }

    @Test
    void calculateIsExactBeyondDoublePrecision() {
        assertEquals("100000000000000000000", ReActTools.calculate("10000000000 * 10000000000"));
        assertEquals("100000000000000000000000", ReActTools.calculate("100000000000 * 1000000000000"));
        assertEquals("9007199254740993", ReActTools.calculate("9007199254740993 + 0"));
        assertEquals("0.3000", ReActTools.calculate("0.1 + 0.2"));
        assertEquals("0", ReActTools.calculate("-0 * 5"));
        assertEquals("3", ReActTools.calculate("1.5 * 2"));
    }

    @Test
    void lookupFindsFactsCaseInsensitively() {
        assertTrue(ReActTools.lookup("EIFFEL TOWER height").contains("330"));
        assertTrue(ReActTools.lookup("Atlantis").startsWith("No fact found"));
    }
}
