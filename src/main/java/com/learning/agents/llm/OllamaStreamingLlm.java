package com.learning.agents.llm;

import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;

import java.time.Duration;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.function.Consumer;

/**
 * Adapts a LangChain4j {@link StreamingChatModel} (such as {@code OllamaStreamingChatModel}) to {@link StreamingLlm}.
 * LangChain4j streams through callbacks on its own thread, so this waits for the final callback, for at most
 * {@link #TIMEOUT} by default, and turns an error callback into an exception.
 */
public final class OllamaStreamingLlm implements StreamingLlm {

    static final Duration TIMEOUT = Duration.ofMinutes(2);

    private final StreamingChatModel model;
    private final Duration timeout;

    public OllamaStreamingLlm(StreamingChatModel model) {
        this(model, TIMEOUT);
    }

    /** Same, with a custom limit on how long to wait for the reply to finish. */
    OllamaStreamingLlm(StreamingChatModel model, Duration timeout) {
        this.model = model;
        this.timeout = timeout;
    }

    /**
     * Streams the reply to {@code onToken} and returns it in full.
     *
     * @throws IllegalStateException if the model reports an error, the wait is interrupted, or the timeout passes
     */
    @Override
    public String stream(String prompt, Consumer<String> onToken) {
        var done = new CompletableFuture<String>();
        model.chat(prompt, new StreamingChatResponseHandler() {
            @Override
            public void onPartialResponse(String partial) {
                onToken.accept(partial);
            }

            @Override
            public void onCompleteResponse(ChatResponse response) {
                done.complete(response.aiMessage().text());
            }

            @Override
            public void onError(Throwable error) {
                done.completeExceptionally(error);
            }
        });
        try {
            return done.get(timeout.toMillis(), TimeUnit.MILLISECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("interrupted while streaming", e);
        } catch (ExecutionException | TimeoutException e) {
            throw new IllegalStateException("streaming failed: " + e.getMessage(), e);
        }
    }
}
