package com.learning.agents.chat;

import com.learning.agents.MainRunner;
import com.learning.agents.chat.StructuredOutputAgent.ReviewSummary;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class StructuredOutputAgentTest {

    @Test
    void repairsAnInvalidReplyUsingTheValidationError() {
        List<String> prompts = new ArrayList<>();
        var result = StructuredOutputAgent.extract(prompt -> {
            prompts.add(prompt);
            return StructuredOutputAgent.placeholder().apply(prompt);
        }, "great shoes");

        assertEquals(new ReviewSummary("TrailRunner 2 shoes", 4, "positive"), result.value().orElseThrow());
        assertEquals(2, result.attempts().size());
        assertTrue(prompts.get(1).contains("\"rating\" must be between 1 and 5, got 8"));
    }

    @Test
    void repairPromptStillContainsTheReviewAndSchema() {
        List<String> prompts = new ArrayList<>();
        StructuredOutputAgent.extract(prompt -> {
            prompts.add(prompt);
            return "{}";
        }, "The Kettle 3000 is loud");

        assertTrue(prompts.get(1).contains("Review: <<<The Kettle 3000 is loud>>>"));
        assertTrue(prompts.get(1).contains("\"rating\": integer 1-5"));
        assertTrue(prompts.get(1).contains("Your previous reply was invalid"));
    }

    @Test
    void wrongTypesAreRejectedNotCoerced() {
        assertThrows(IllegalArgumentException.class, () -> StructuredOutputAgent.parse("{\"product\": \"p\", \"rating\": 4.9, \"sentiment\": \"neutral\"}"));
        assertThrows(IllegalArgumentException.class, () -> StructuredOutputAgent.parse("{\"product\": \"p\", \"rating\": \"5\", \"sentiment\": \"neutral\"}"));
    }

    @Test
    void givesUpAfterMaxAttempts() {
        var result = StructuredOutputAgent.extract(prompt -> "not json at all", "review");

        assertTrue(result.value().isEmpty());
        assertEquals(StructuredOutputAgent.MAX_ATTEMPTS, result.attempts().size());
    }

    @Test
    void parseAcceptsFencedJsonAndIgnoresExtraFields() {
        var summary = StructuredOutputAgent.parse("""
                Sure! ```json
                {"product": "Kettle", "rating": 2, "sentiment": "negative", "extra": true}
                ```""");
        assertEquals(new ReviewSummary("Kettle", 2, "negative"), summary);
    }

    @Test
    void parseRejectsEachKindOfInvalidValue() {
        assertThrows(IllegalArgumentException.class, () -> StructuredOutputAgent.parse("{\"rating\": 3, \"sentiment\": \"neutral\"}"));
        assertThrows(IllegalArgumentException.class, () -> StructuredOutputAgent.parse("{\"product\": \" \", \"rating\": 3, \"sentiment\": \"neutral\"}"));
        assertThrows(IllegalArgumentException.class, () -> StructuredOutputAgent.parse("{\"product\": \"p\", \"rating\": 0, \"sentiment\": \"neutral\"}"));
        assertThrows(IllegalArgumentException.class, () -> StructuredOutputAgent.parse("{\"product\": \"p\", \"rating\": 3, \"sentiment\": \"meh\"}"));
        assertThrows(IllegalArgumentException.class, () -> StructuredOutputAgent.parse("{\"product\": \"p\", \"rating\": 3}"));
        assertThrows(IllegalArgumentException.class, () -> StructuredOutputAgent.parse("{\"product\": \"p\", \"sentiment\": \"neutral\"}"));
    }

    @Test
    void stripFencesLeavesNonJsonAlone() {
        assertEquals("no braces", StructuredOutputAgent.stripFences("  no braces "));
    }

    @Test
    void mainShowsTheRepair() throws Exception {
        String out = MainRunner.run(StructuredOutputAgent::main);
        assertTrue(out.contains("Attempt 2") && out.contains("rating=4"));
        assertTrue(MainRunner.run(StructuredOutputAgent::main, "x").contains("Parsed"));
    }
}
