package com.learning.agents;

import dev.langchain4j.model.ollama.OllamaChatModel;

import java.time.Duration;

/**
 * Central place for building Ollama-backed models.
 * <p>
 * Settings come from environment variables so examples run unchanged on any machine:
 * <ul>
 *   <li>{@code OLLAMA_BASE_URL} – default {@value #DEFAULT_BASE_URL}</li>
 *   <li>{@code OLLAMA_MODEL} – default {@value #DEFAULT_MODEL} (must support tool calling for the tool/graph examples)</li>
 * </ul>
 */
public final class OllamaConfig {

    public static final String DEFAULT_BASE_URL = "http://localhost:11434";
    public static final String DEFAULT_MODEL = "llama3.1";

    private OllamaConfig() {
    }

    public static String baseUrl() {
        return env("OLLAMA_BASE_URL", DEFAULT_BASE_URL);
    }

    public static String modelName() {
        return env("OLLAMA_MODEL", DEFAULT_MODEL);
    }

    public static OllamaChatModel chatModel() {
        return OllamaChatModel.builder()
                .baseUrl(baseUrl())
                .modelName(modelName())
                .temperature(0.2)
                .timeout(Duration.ofMinutes(2))
                .logRequests(false)
                .logResponses(false)
                .build();
    }

    private static String env(String name, String defaultValue) {
        String value = System.getenv(name);
        return value == null || value.isBlank() ? defaultValue : value;
    }
}
