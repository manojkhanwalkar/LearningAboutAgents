package com.learning.agents.llm;

import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.PartialResponse;
import dev.langchain4j.model.chat.response.PartialResponseContext;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import dev.langchain4j.model.chat.response.StreamingHandle;

import java.time.Duration;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

/**
 * Adapts a LangChain4j {@link StreamingChatModel} (such as {@code OllamaStreamingChatModel}) to {@link StreamingLlm}.
 * LangChain4j streams through callbacks on its own thread, so this waits for the final callback, for at most
 * {@link #TIMEOUT} by default, and turns an error callback into an exception.
 * <p>
 * Giving up must also <em>stop</em> the stream: on a timeout or an interrupt the request is cancelled through
 * LangChain4j's {@link StreamingHandle}, and any tokens that still arrive are dropped instead of reaching
 * {@code onToken} after the caller has moved on.
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
     * Streams the reply to {@code onToken} and returns it in full ({@code ""} if the model sent no text).
     *
     * @throws IllegalStateException if the model reports an error, the wait is interrupted, or the timeout passes;
     *                               in the last two cases the request is cancelled first
     */
    @Override
    public String stream(String prompt, Consumer<String> onToken) {
        var done = new CompletableFuture<String>();
        var handle = new AtomicReference<StreamingHandle>();
        var abandoned = new AtomicBoolean();
        model.chat(prompt, new StreamingChatResponseHandler() {
            @Override
            public void onPartialResponse(PartialResponse partial, PartialResponseContext context) {
                if (handle.compareAndSet(null, context.streamingHandle()) && abandoned.get()) {
                    // The caller gave up before the first token brought this handle, so abandon() found none to cancel.
                    context.streamingHandle().cancel();
                    return;
                }
                deliver(partial.text());
            }

            @Override
            public void onPartialResponse(String partial) {
                deliver(partial);
            }

            private void deliver(String partial) {
                if (!abandoned.get()) {
                    onToken.accept(partial);
                }
            }

            @Override
            public void onCompleteResponse(ChatResponse response) {
                done.complete(Objects.requireNonNullElse(response.aiMessage().text(), ""));
            }

            @Override
            public void onError(Throwable error) {
                done.completeExceptionally(error);
            }
        });
        try {
            return done.get(timeout.toMillis(), TimeUnit.MILLISECONDS);
        } catch (InterruptedException e) {
            abandon(abandoned, handle);
            Thread.currentThread().interrupt();
            throw new IllegalStateException("interrupted while streaming", e);
        } catch (TimeoutException e) {
            abandon(abandoned, handle);
            throw new IllegalStateException("no complete reply within " + timeout, e);
        } catch (ExecutionException e) {
            throw new IllegalStateException("streaming failed: " + e.getCause().getMessage(), e.getCause());
        }
    }

    /**
     * Stops forwarding tokens and asks the client to cancel the request, if it has given us a handle yet. If not, the
     * callback that brings the handle cancels it: it sets the handle before reading {@code abandoned}, and this sets
     * {@code abandoned} before reading the handle, so at least one side always sees the other.
     */
    private static void abandon(AtomicBoolean abandoned, AtomicReference<StreamingHandle> handle) {
        abandoned.set(true);
        StreamingHandle h = handle.get();
        if (h != null) {
            h.cancel();
        }
    }
}
