package com.learning.agents.llm;

import org.junit.jupiter.api.Test;

import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LlmProviderTest {

    private final Function<String, String> placeholder = prompt -> "placeholder";
    private final Function<String, String> ollama = prompt -> "ollama";

    @Test
    void defaultsToPlaceholderWhenUnsetOrBlank() {
        assertSame(placeholder, LlmProvider.resolve(null, placeholder, () -> ollama));
        assertSame(placeholder, LlmProvider.resolve("  ", placeholder, () -> ollama));
        assertSame(placeholder, LlmProvider.resolve("Placeholder", placeholder, () -> ollama));
    }

    @Test
    void selectsOllamaCaseInsensitively() {
        assertSame(ollama, LlmProvider.resolve(" OLLAMA ", placeholder, () -> ollama));
    }

    @Test
    void rejectsUnknownBackend() {
        var e = assertThrows(IllegalArgumentException.class,
                () -> LlmProvider.resolve("openai", placeholder, () -> ollama));
        assertTrue(e.getMessage().contains("openai"));
    }

    @Test
    void publicResolveReturnsAWorkingFunction() {
        // Uses the real environment: the placeholder when LLM_BACKEND is unset. Never calls the model here.
        assertTrue(LlmProvider.resolve(placeholder) != null);
    }

    @Test
    void resolveEmbeddingsUsesThePlaceholderByDefault() {
        var placeholderEmbeddings = new HashingEmbeddingModel();
        String backend = System.getenv(LlmProvider.ENV);
        if (backend == null || backend.isBlank()) {
            assertSame(placeholderEmbeddings, LlmProvider.resolveEmbeddings(placeholderEmbeddings));
        }
    }

    @Test
    void scriptedLlmMatchesFirstRuleAndFallsBack() {
        var llm = ScriptedLlm.create()
                .when("alpha", "first")
                .when(prompt -> prompt.contains("alpha") || prompt.contains("beta"), prompt -> "second:" + prompt)
                .otherwise("none");

        assertEquals("first", llm.apply("alpha beta"));
        assertEquals("second:beta", llm.apply("beta"));
        assertEquals("none", llm.apply("gamma"));
        assertEquals("(placeholder reply)", ScriptedLlm.create().apply("anything"));
    }

    @Test
    void scriptedLlmIsImmutable() {
        var base = ScriptedLlm.create().otherwise("base");
        var extended = base.when("x", "extended");

        assertEquals("base", base.apply("x"));
        assertEquals("extended", extended.apply("x"));
    }
}
