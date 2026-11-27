package com.learning.agents;

import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.net.http.HttpClient;
import java.time.Duration;
import java.util.ArrayDeque;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

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

    @Test
    void connectingFailsFastWhileReplyingMayTakeLong() throws Exception {
        // LangChain4j's builder timeout(...) sets the connect timeout too, so an unreachable host would hold each of
        // the three attempts for the full reply timeout. Connecting gets its own short limit instead.
        Map<Object, Duration> replyTimeouts = Map.of(
                OllamaConfig.chatModel(), OllamaConfig.CHAT_TIMEOUT,
                OllamaConfig.streamingChatModel(), OllamaConfig.CHAT_TIMEOUT,
                OllamaConfig.embeddingModel(), OllamaConfig.EMBEDDING_TIMEOUT);
        for (var entry : replyTimeouts.entrySet()) {
            String name = entry.getKey().getClass().getSimpleName();
            HttpClient http = find(entry.getKey(), HttpClient.class).orElseThrow();
            assertEquals(Optional.of(OllamaConfig.CONNECT_TIMEOUT), http.connectTimeout(), name);
            assertEquals(entry.getValue(), find(entry.getKey(), Duration.class).orElseThrow(), name);   // the read timeout
        }
    }

    /** Finds the first value of {@code type} reachable through the fields of LangChain4j's own objects. */
    private static <T> Optional<T> find(Object root, Class<T> type) throws IllegalAccessException {
        Set<Object> seen = Collections.newSetFromMap(new IdentityHashMap<>());
        var queue = new ArrayDeque<>(java.util.List.of(root));
        while (!queue.isEmpty()) {
            Object current = queue.poll();
            if (!seen.add(current)) {
                continue;
            }
            for (Class<?> c = current.getClass(); c != null && c.getName().startsWith("dev.langchain4j."); c = c.getSuperclass()) {
                for (Field field : c.getDeclaredFields()) {
                    if (Modifier.isStatic(field.getModifiers()) || field.getType().isPrimitive()) {
                        continue;
                    }
                    field.setAccessible(true);
                    Object value = field.get(current);
                    if (type.isInstance(value)) {
                        return Optional.of(type.cast(value));
                    }
                    if (value != null && value.getClass().getName().startsWith("dev.langchain4j.")) {
                        queue.add(value);
                    }
                }
            }
        }
        return Optional.empty();
    }
}
