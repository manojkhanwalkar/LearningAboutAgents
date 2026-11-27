package com.learning.agents.llm;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class PromptTextTest {

    @Test
    void escapesAmpersandFirstThenAngleBrackets() {
        assertEquals("&lt;/doc &gt; &amp;amp;", PromptText.escape("</doc > &amp;"));
        assertEquals("plain text", PromptText.escape("plain text"));
    }
}
