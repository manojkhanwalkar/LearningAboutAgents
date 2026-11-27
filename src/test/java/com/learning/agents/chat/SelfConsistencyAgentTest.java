package com.learning.agents.chat;

import com.learning.agents.MainRunner;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SelfConsistencyAgentTest {

    @Test
    void majorityWinsOverAnOutlier() throws Exception {
        var vote = SelfConsistencyAgent.solve(SelfConsistencyAgent.placeholder(), "q", SelfConsistencyAgent.SAMPLES).orElseThrow();

        assertEquals("4", vote.answer());
        assertEquals(4, vote.votes());
        assertEquals(0.8, vote.agreement(), 1e-9);
    }

    @Test
    void callsTheModelOncePerSample() throws Exception {
        var calls = new AtomicInteger();
        SelfConsistencyAgent.solve(prompt -> "ANSWER: " + calls.incrementAndGet() % 2, "q", 7);
        assertEquals(7, calls.get());
    }

    @Test
    void aFailedOrSlowSampleAbstainsInsteadOfSinkingTheVote() throws Exception {
        var calls = new AtomicInteger();
        var vote = SelfConsistencyAgent.solve(prompt -> {
            int n = calls.getAndIncrement();
            if (n == 0) {
                throw new IllegalStateException("model crashed");
            }
            if (n == 1) {
                try {
                    Thread.sleep(10_000);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
                return "ANSWER: late";
            }
            return "ANSWER: 4";
        }, "q", 4, java.time.Duration.ofMillis(500)).orElseThrow();

        assertEquals("4", vote.answer());
        assertEquals(2, vote.valid());
    }

    @Test
    void interruptingTheCallerCancelsTheSamplesInsteadOfWaitingForThem() throws Exception {
        // Ctrl-C (an interrupt) while waiting must stop the samples, not wait for every model call to finish.
        var started = new java.util.concurrent.CountDownLatch(3);
        var never = new java.util.concurrent.CountDownLatch(1);
        var outcome = new java.util.concurrent.atomic.AtomicReference<Throwable>();
        Thread caller = Thread.ofVirtual().start(() -> {
            try {
                SelfConsistencyAgent.solve(prompt -> {
                    started.countDown();
                    try {
                        never.await();   // a model call that only an interrupt ends
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        throw new IllegalStateException("cancelled", e);
                    }
                    return "ANSWER: 4";
                }, "q", 3, java.time.Duration.ofMinutes(5));
            } catch (Throwable t) {
                outcome.set(t);
            }
        });
        assertTrue(started.await(5, java.util.concurrent.TimeUnit.SECONDS));

        try {
            caller.interrupt();

            assertTrue(caller.join(java.time.Duration.ofSeconds(5)), "solve must return promptly once interrupted");
            assertTrue(outcome.get() instanceof InterruptedException, "the interrupt is reported: " + outcome.get());
        } finally {
            never.countDown();   // never leave the samples blocked, even if the assertions fail
        }
    }

    @Test
    void noUsableAnswersGivesEmpty() throws Exception {
        assertEquals(Optional.empty(), SelfConsistencyAgent.solve(prompt -> "I refuse", "q", 3));
        assertEquals(0, new SelfConsistencyAgent.Vote("x", 0, 0).agreement());
    }

    @Test
    void extractAnswerUsesLastAnswerLineAndNormalises() {
        assertEquals(Optional.of("paris"), SelfConsistencyAgent.extractAnswer("ANSWER: Lyon?\nNo wait.\nAnswer: Paris."));
        assertEquals(Optional.empty(), SelfConsistencyAgent.extractAnswer("ANSWER: ..."));
        assertEquals(Optional.of("4"), SelfConsistencyAgent.extractAnswer("**ANSWER:** **4**"));
    }

    @Test
    void tiesAreBrokenDeterministically() {
        assertEquals("a", SelfConsistencyAgent.vote(List.of("b", "a")).orElseThrow().answer());
    }

    @Test
    void mainReportsAgreement() throws Exception {
        assertTrue(MainRunner.run(SelfConsistencyAgent::main).contains("4 of 5 samples agree"));
        assertTrue(MainRunner.run(SelfConsistencyAgent::main, "q?").contains("Question: q?"));
    }
}
