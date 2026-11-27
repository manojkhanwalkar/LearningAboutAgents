package com.learning.agents.llm;

import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;

class RepliesTest {

    @Test
    void findsDecoratedLabelsOnAnyLine() {
        assertEquals(Optional.of("fine"), Replies.after("**ANSWER:** fine", "ANSWER:"));
        assertEquals(Optional.of("Paris\nmore"), Replies.after("Let me think.\n> answer: Paris\nmore", "ANSWER:"));
        assertEquals(Optional.of("Paris"), Replies.lineAfter("Let me think.\nANSWER: Paris\nmore", "ANSWER:"));
        assertEquals(Optional.of("-5"), Replies.after("ANSWER: -5", "ANSWER:"));
        assertEquals(Optional.of("(a) first"), Replies.after("ANSWER: (a) first", "ANSWER:"));
    }

    @Test
    void missingOrEmptyLabelsGiveEmpty() {
        assertEquals(Optional.empty(), Replies.after("no label here", "ANSWER:"));
        assertEquals(Optional.empty(), Replies.after("ANSWER:   ", "ANSWER:"));
        assertEquals(Optional.empty(), Replies.after("", "ANSWER:"));
    }

    @Test
    void longBlankRepliesAreScannedInLinearTime() {
        String stuck = "\n".repeat(200_000) + "  ";
        // The old (?m)^\W*ANSWER: pattern took seconds on 20k newlines; this must stay well under a second for 200k.
        assertTimeoutPreemptively(Duration.ofSeconds(1), () -> assertEquals(Optional.empty(), Replies.after(stuck, "ANSWER:")));
    }
}
