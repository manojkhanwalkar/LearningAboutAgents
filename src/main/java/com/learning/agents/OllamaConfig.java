package com.learning.agents;

import dev.langchain4j.http.client.HttpClientBuilder;
import dev.langchain4j.http.client.HttpClientBuilderLoader;
import dev.langchain4j.model.ollama.OllamaChatModel;
import dev.langchain4j.model.ollama.OllamaEmbeddingModel;
import dev.langchain4j.model.ollama.OllamaStreamingChatModel;

import java.time.Duration;

/**
 * Central place for building Ollama-backed models.
 * <p>
 * Settings come from environment variables so examples run unchanged on any machine:
 * <ul>
 *   <li>{@code OLLAMA_BASE_URL} – default {@value #DEFAULT_BASE_URL}</li>
 *   <li>{@code OLLAMA_MODEL} – default {@value #DEFAULT_MODEL} (must support tool calling for {@code ToolCallingAgent})</li>
 *   <li>{@code OLLAMA_EMBEDDING_MODEL} – default {@value #DEFAULT_EMBEDDING_MODEL}, used by the embedding samples</li>
 * </ul>
 */
public final class OllamaConfig {

    public static final String DEFAULT_BASE_URL = "http://localhost:11434";
    public static final String DEFAULT_MODEL = "llama3.1";
    public static final String DEFAULT_EMBEDDING_MODEL = "nomic-embed-text";

    /** How long to wait for a connection to Ollama; separate from the reply timeouts, which must allow slow generation. */
    static final Duration CONNECT_TIMEOUT = Duration.ofSeconds(10);
    static final Duration CHAT_TIMEOUT = Duration.ofMinutes(2);
    static final Duration EMBEDDING_TIMEOUT = Duration.ofMinutes(1);

    private OllamaConfig() {
    }

    public static String baseUrl() {
        return env("OLLAMA_BASE_URL", DEFAULT_BASE_URL);
    }

    public static String modelName() {
        return env("OLLAMA_MODEL", DEFAULT_MODEL);
    }

    public static String embeddingModelName() {
        return env("OLLAMA_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL);
    }

    public static OllamaChatModel chatModel() {
        return chatModel(modelName());
    }

    /** A chat model like {@link #chatModel()} but for a named model, for samples that use more than one. */
    public static OllamaChatModel chatModel(String modelName) {
        return chatModel(modelName, null);
    }

    /**
     * A chat model for a named model with its own retry count. Pass {@code 0} when the caller retries itself, so a
     * failing call isn't retried by two layers at once; {@code null} keeps LangChain4j's default.
     */
    public static OllamaChatModel chatModel(String modelName, Integer maxRetries) {
        return OllamaChatModel.builder()
                .baseUrl(baseUrl())
                .modelName(modelName)
                .maxRetries(maxRetries)
                .temperature(0.2)
                .httpClientBuilder(http(CHAT_TIMEOUT))
                .logRequests(false)
                .logResponses(false)
                .build();
    }

    /** A streaming chat model with the same settings as {@link #chatModel()}, for samples that print tokens as they arrive. */
    public static OllamaStreamingChatModel streamingChatModel() {
        return OllamaStreamingChatModel.builder()
                .baseUrl(baseUrl())
                .modelName(modelName())
                .temperature(0.2)
                .httpClientBuilder(http(CHAT_TIMEOUT))
                .build();
    }

    /** The embedding model the embedding samples (RAG, memory) use when {@code LLM_BACKEND=ollama}. */
    public static OllamaEmbeddingModel embeddingModel() {
        return OllamaEmbeddingModel.builder()
                .baseUrl(baseUrl())
                .modelName(embeddingModelName())
                .httpClientBuilder(http(EMBEDDING_TIMEOUT))
                .build();
    }

    /**
     * An HTTP client with a short connect timeout and a long read timeout. LangChain4j's {@code timeout(...)} builder
     * option sets both to the same value, so an unreachable host would hold every attempt (three, with the default
     * retries) for the full reply time instead of failing in seconds.
     */
    private static HttpClientBuilder http(Duration readTimeout) {
        return HttpClientBuilderLoader.loadHttpClientBuilder()
                .connectTimeout(CONNECT_TIMEOUT)
                .readTimeout(readTimeout);
    }

    private static String env(String name, String defaultValue) {
        String value = System.getenv(name);
        return value == null || value.isBlank() ? defaultValue : value;
    }
}
