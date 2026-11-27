package com.learning.agents.llm;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class StreamingLlmTest {

    /** A fake LangChain4j streaming model whose behaviour each test scripts. */
    private static StreamingChatModel fake(java.util.function.Consumer<StreamingChatResponseHandler> script) {
        return new StreamingChatModel() {
            @Override
            public void doChat(ChatRequest request, StreamingChatResponseHandler handler) {
                // Like a real client, answer on another thread.
                Thread.ofVirtual().start(() -> script.accept(handler));
            }
        };
    }

    @Test
    void placeholderStreamsWordByWordAndReturnsTheWholeReply() {
        List<String> tokens = new ArrayList<>();
        String reply = StreamingLlm.fromReplies(prompt -> "one two three").stream("p", tokens::add);

        assertEquals("one two three", reply);
        assertEquals(List.of("one ", "two ", "three"), tokens);
    }

    @Test
    void ollamaAdapterForwardsTokensAndWaitsForCompletion() {
        List<String> tokens = new ArrayList<>();
        var llm = new OllamaStreamingLlm(fake(handler -> {
            handler.onPartialResponse("Hel");
            handler.onPartialResponse("lo");
            handler.onCompleteResponse(ChatResponse.builder().aiMessage(AiMessage.from("Hello")).build());
        }));

        assertEquals("Hello", llm.stream("hi", tokens::add));
        assertEquals(List.of("Hel", "lo"), tokens);
    }

    @Test
    void anErrorCallbackBecomesAnException() {
        var llm = new OllamaStreamingLlm(fake(handler -> handler.onError(new RuntimeException("model not found"))));
        var e = assertThrows(IllegalStateException.class, () -> llm.stream("hi", t -> { }));
        assertTrue(e.getMessage().contains("model not found"));
    }

    @Test
    void aTimedOutStreamIsCancelledAndLateTokensAreDropped() throws Exception {
        var cancelled = new java.util.concurrent.atomic.AtomicBoolean();
        var keepGoing = new java.util.concurrent.CountDownLatch(1);
        List<String> tokens = java.util.Collections.synchronizedList(new ArrayList<>());
        var llm = new OllamaStreamingLlm(fake(handler -> {
            var context = new dev.langchain4j.model.chat.response.PartialResponseContext(
                    new dev.langchain4j.model.chat.response.StreamingHandle() {
                        @Override
                        public void cancel() {
                            cancelled.set(true);
                        }

                        @Override
                        public boolean isCancelled() {
                            return cancelled.get();
                        }
                    });
            handler.onPartialResponse(new dev.langchain4j.model.chat.response.PartialResponse("early "), context);
            try {
                keepGoing.await();   // stall past the timeout, then keep "generating"
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            handler.onPartialResponse(new dev.langchain4j.model.chat.response.PartialResponse("late"), context);
        }), Duration.ofMillis(100));

        var e = assertThrows(IllegalStateException.class, () -> llm.stream("hi", tokens::add));
        keepGoing.countDown();
        Thread.sleep(100);

        assertTrue(e.getMessage().contains("no complete reply within"));
        assertTrue(cancelled.get(), "the request must be cancelled");
        assertEquals(List.of("early "), tokens);
    }

    @Test
    void anEmptyReplyBecomesAnEmptyString() {
        // A reply made only of tool calls has no text: aiMessage().text() is null.
        var toolOnly = AiMessage.from(List.of(dev.langchain4j.agent.tool.ToolExecutionRequest.builder().name("t").arguments("{}").build()));
        var llm = new OllamaStreamingLlm(fake(handler ->
                handler.onCompleteResponse(ChatResponse.builder().aiMessage(toolOnly).build())), Duration.ofSeconds(5));
        assertEquals("", llm.stream("hi", t -> { }));
    }

    @Test
    void interruptionIsReportedAndTheFlagRestored() {
        var llm = new OllamaStreamingLlm(fake(handler -> { }), Duration.ofSeconds(10));
        Thread.currentThread().interrupt();
        try {
            assertThrows(IllegalStateException.class, () -> llm.stream("hi", t -> { }));
            assertTrue(Thread.interrupted(), "interrupt flag should be restored");
        } finally {
            Thread.interrupted();
        }
    }

    @Test
    void resolveStreamingDefaultsToThePlaceholder() {
        StreamingLlm placeholder = (prompt, onToken) -> "x";
        StreamingLlm other = (prompt, onToken) -> "y";
        assertSame(placeholder, LlmProvider.resolve(null, placeholder, () -> other));
        assertSame(other, LlmProvider.resolve("ollama", placeholder, () -> other));
        String backend = System.getenv(LlmProvider.ENV);
        if (backend == null || backend.isBlank()) {
            assertSame(placeholder, LlmProvider.resolveStreaming(placeholder));
        }
        assertNotNull(com.learning.agents.OllamaConfig.streamingChatModel());
    }
}
