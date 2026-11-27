package com.learning.agents.llm;

import java.util.function.Consumer;

/**
 * A model that delivers its reply piece by piece. {@code onToken} is called with each piece as it arrives, and the
 * method returns the whole reply once the model is done. The streaming counterpart of the
 * {@code Function<String, String>} the other samples use.
 */
@FunctionalInterface
public interface StreamingLlm {

    /**
     * Sends {@code prompt} and streams the reply.
     *
     * @param onToken called with each piece of the reply, in order, on the caller's thread or the client's
     * @return the complete reply
     */
    String stream(String prompt, Consumer<String> onToken);

    /** A placeholder that "streams" a canned reply word by word, so streaming code can run without a model. */
    static StreamingLlm fromReplies(java.util.function.Function<String, String> replies) {
        return (prompt, onToken) -> {
            String reply = replies.apply(prompt);
            for (String piece : reply.split("(?<= )")) {
                onToken.accept(piece);
            }
            return reply;
        };
    }
}
