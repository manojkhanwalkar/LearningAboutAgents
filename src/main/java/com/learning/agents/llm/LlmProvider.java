package com.learning.agents.llm;

import com.learning.agents.OllamaConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Locale;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * Chooses the LLM for a sample: the sample's scripted placeholder, or a real Ollama model.
 * <p>
 * Every sample is written against a plain {@code Function<String, String>} (prompt in, reply out), so moving from
 * the placeholder to a real model is a configuration change, not a code change. The environment variable
 * {@value #ENV} picks the backend:
 * <ul>
 *   <li>unset, blank or {@code placeholder}: the scripted replies (no server needed)</li>
 *   <li>{@code ollama}: {@link OllamaConfig#chatModel()}, so {@code OLLAMA_BASE_URL} and {@code OLLAMA_MODEL}
 *       apply (for example {@code OLLAMA_MODEL=gemma3})</li>
 * </ul>
 */
public final class LlmProvider {

    public static final String ENV = "LLM_BACKEND";

    private static final Logger log = LoggerFactory.getLogger(LlmProvider.class);

    private LlmProvider() {
    }

    /** Returns the backend named by {@value #ENV}, falling back to {@code placeholder}. */
    public static Function<String, String> resolve(Function<String, String> placeholder) {
        return resolve(System.getenv(ENV), placeholder, () -> OllamaConfig.chatModel()::chat);
    }

    /**
     * Same as {@link #resolve(Function)} with the backend name and the Ollama factory passed in, so it can be tested.
     *
     * @throws IllegalArgumentException if {@code backend} is not a known name
     */
    static Function<String, String> resolve(String backend, Function<String, String> placeholder,
                                            Supplier<Function<String, String>> ollama) {
        String name = backend == null || backend.isBlank() ? "placeholder" : backend.strip().toLowerCase(Locale.ROOT);
        return switch (name) {
            case "placeholder" -> {
                log.info("Using the scripted placeholder LLM. Set {}=ollama to call a real model.", ENV);
                yield placeholder;
            }
            case "ollama" -> {
                log.info("Using Ollama model '{}' at {}", OllamaConfig.modelName(), OllamaConfig.baseUrl());
                yield ollama.get();
            }
            default -> throw new IllegalArgumentException(
                    "Unknown %s '%s'. Use 'placeholder' or 'ollama'.".formatted(ENV, backend));
        };
    }
}
