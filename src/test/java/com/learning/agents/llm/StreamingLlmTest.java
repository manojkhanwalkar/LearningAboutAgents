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
    void aReplyThatNeverFinishesTimesOut() {
        var llm = new OllamaStreamingLlm(fake(handler -> handler.onPartialResponse("...")), Duration.ofMillis(100));
        assertThrows(IllegalStateException.class, () -> llm.stream("hi", t -> { }));
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
        String backend = System.getenv(LlmProvider.ENV);
        if (backend == null || backend.isBlank()) {
            assertSame(placeholder, LlmProvider.resolveStreaming(placeholder));
        }
        assertNotNull(com.learning.agents.OllamaConfig.streamingChatModel());
    }
}
