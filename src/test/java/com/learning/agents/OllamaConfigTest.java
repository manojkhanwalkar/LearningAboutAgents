package com.learning.agents;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;

/**
 * Building an Ollama client opens no connection, so the factories can be checked without a server.
 */
class OllamaConfigTest {

    @Test
    void settingsFallBackToDefaultsAndAreNeverBlank() {
        assertFalse(OllamaConfig.baseUrl().isBlank());
        assertFalse(OllamaConfig.modelName().isBlank());
        assertFalse(OllamaConfig.embeddingModelName().isBlank());
        if (System.getenv("OLLAMA_EMBEDDING_MODEL") == null) {
            assertEquals(OllamaConfig.DEFAULT_EMBEDDING_MODEL, OllamaConfig.embeddingModelName());
        }
    }

    @Test
    void factoriesBuildClientsWithoutConnecting() {
        assertNotNull(OllamaConfig.chatModel());
        assertNotNull(OllamaConfig.chatModel("gemma3"));
        assertNotNull(OllamaConfig.embeddingModel());
    }
}
