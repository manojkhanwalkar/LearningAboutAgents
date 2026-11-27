package com.learning.agents.llm;

/**
 * Helpers for putting untrusted text (user messages, documents, stored memories) inside a prompt.
 */
public final class PromptText {

    private PromptText() {
    }

    /**
     * Escapes {@code &}, {@code <} and {@code >}, so untrusted text placed inside a tag such as {@code <doc>} or
     * {@code <memory>} can't close it early and smuggle in instructions. The model still reads it fine.
     */
    public static String escape(String text) {
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;");
    }
}
