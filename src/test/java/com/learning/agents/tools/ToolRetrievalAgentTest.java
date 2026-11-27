package com.learning.agents.tools;

import com.learning.agents.MainRunner;
import com.learning.agents.llm.HashingEmbeddingModel;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ToolRetrievalAgentTest {

    private ToolRetrievalAgent agent(java.util.function.Function<String, String> llm) {
        return new ToolRetrievalAgent(ToolRetrievalAgent.CATALOG, new HashingEmbeddingModel(), llm);
    }

    @Test
    void retrievalPutsTheRelevantToolFirstAndShowsOnlyTopK() {
        var shown = agent(p -> "NONE").retrieve("convert kilometres to miles", ToolRetrievalAgent.TOP_K);
        assertEquals(ToolRetrievalAgent.TOP_K, shown.size());
        assertEquals("km_to_miles", shown.getFirst().name());
        assertTrue(agent(p -> "NONE").retrieve("  ", 3).isEmpty());
    }

    @Test
    void theModelOnlySeesTheRetrievedTools() {
        List<String> prompts = new ArrayList<>();
        agent(p -> {
            prompts.add(p);
            return "NONE";
        }).ask("Is racecar a palindrome?");

        assertTrue(prompts.getFirst().contains("- palindrome_check"));
        assertFalse(prompts.getFirst().contains("- kg_to_pounds"));
    }

    @Test
    void callsTheChosenToolAndReturnsItsResult() {
        var outcome = agent(ToolRetrievalAgent.placeholder()).ask("How far is 42.195 kilometres in miles?");
        assertEquals(Optional.of("km_to_miles(42.195)"), outcome.call());
        assertEquals("26.22", outcome.result());
    }

    @Test
    void aToolThatWasNotShownIsRefused() {
        var outcome = agent(p -> "CALL kg_to_pounds(3)").ask("Is racecar a palindrome?");
        assertTrue(outcome.result().startsWith("error: tool 'kg_to_pounds' was not offered"));
    }

    @Test
    void badInputAndNoneAreHandled() {
        var bad = agent(p -> "CALL km_to_miles(far)").ask("convert kilometres to miles");
        assertTrue(bad.result().startsWith("error: bad input for km_to_miles"));
        var none = agent(p -> "NONE").ask("convert kilometres to miles");
        assertTrue(none.call().isEmpty() && none.result().startsWith("No suitable tool"));
    }

    @Test
    void inputRunsToTheLastParenthesisAndNumbersIgnoreTheLocale() {
        var outcome = agent(p -> "Sure!\nCALL reverse_text(a) b)").ask("Reverse the characters of a piece of text");
        assertEquals("b )a", outcome.result());

        var original = java.util.Locale.getDefault();
        try {
            java.util.Locale.setDefault(java.util.Locale.FRANCE);
            assertEquals("26.22", agent(ToolRetrievalAgent.placeholder()).ask("How far is 42.195 kilometres in miles?").result());
        } finally {
            java.util.Locale.setDefault(original);
        }
    }

    @Test
    void everyCatalogToolWorks() {
        var byName = new java.util.HashMap<String, ToolRetrievalAgent.Tool>();
        ToolRetrievalAgent.CATALOG.forEach(t -> byName.put(t.name(), t));
        assertEquals("3", byName.get("word_count").run().apply("a b c"));
        assertEquals("0", byName.get("word_count").run().apply(" "));
        assertEquals("cba", byName.get("reverse_text").run().apply("abc"));
        assertEquals("ABC", byName.get("uppercase").run().apply("abc"));
        assertEquals("6.61", byName.get("kg_to_pounds").run().apply("3"));
        assertEquals("212.0", byName.get("celsius_to_fahrenheit").run().apply("100"));
        assertEquals("true", byName.get("is_leap_year").run().apply("2024"));
        assertEquals("2", byName.get("vowel_count").run().apply("hello"));
        assertEquals("1.4142", byName.get("square_root").run().apply("2"));
        assertEquals("12.00", byName.get("percent_of").run().apply("15% of 80"));
        assertTrue(byName.get("percent_of").run().apply("lots").startsWith("error"));
    }

    @Test
    void mainAnswersAllThreeQuestions() throws Exception {
        String out = MainRunner.run(ToolRetrievalAgent::main);
        assertTrue(out.contains("→ 26.22") && out.contains("→ FRIDAY") && out.contains("→ true"));
    }
}
