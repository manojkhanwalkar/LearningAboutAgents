package com.learning.agents.graph;

import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
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
                .invoke(Map.of(ReActState.QUESTION, "How many times taller is the Burj Khalifa than the Eiffel Tower?"))
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
                .invoke(Map.of(ReActState.QUESTION, "Where is Atlantis?"))
                .orElseThrow();

        assertTrue(result.answer().isEmpty());
        assertEquals(ReActAgentGraph.MAX_STEPS, calls.get());
    }

    @Test
    void malformedReplyProducesCorrectiveObservation() throws Exception {
        ReActState result = ReActAgentGraph.build(prompt -> prompt.contains("Invalid action")
                        ? "Final Answer: 42"
                        : "I think the answer is 42.")
                .invoke(Map.of(ReActState.QUESTION, "What is the answer?"))
                .orElseThrow();

        assertEquals("42", result.answer().orElseThrow());
        assertTrue(result.transcript().get(1).startsWith("Observation: Invalid action"));
    }

    @Test
    void parseExtractsThoughtAndAction() {
        var step = ReActAgentGraph.parse("Thought: need data\nAction: lookup[ Mount Everest ]");
        assertEquals("need data", step.thought());
        assertEquals("lookup", step.action());
        assertEquals("Mount Everest", step.input());
        assertNull(step.answer());
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
        assertEquals("Error: number is out of range", ReActTools.calculate(huge + " + 1"));
        String big = "1" + "0".repeat(300);
        assertEquals("Error: result is out of range", ReActTools.calculate(big + " * " + big));
        assertEquals("100000000000000000000", ReActTools.calculate("10000000000 * 10000000000"));
    }

    @Test
    void lookupFindsFactsCaseInsensitively() {
        assertTrue(ReActTools.lookup("EIFFEL TOWER height").contains("330"));
        assertTrue(ReActTools.lookup("Atlantis").startsWith("No fact found"));
    }
}
