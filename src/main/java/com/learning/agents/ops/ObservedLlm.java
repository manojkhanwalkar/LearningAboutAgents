package com.learning.agents.ops;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.List;
import java.util.function.Function;
import java.util.function.LongSupplier;

/**
 * Observability and a token budget for any LLM, as a decorator: wrap the {@code Function<String, String>} a sample
 * already uses, and every call is measured (latency, prompt and reply size, estimated tokens) and checked against a
 * budget. Nothing about the agent changes, which is the point of a decorator.
 * <p>
 * <b>Budget.</b> Before each call, the prompt's estimated tokens are checked against what is left; if the call would
 * overspend, it is refused with {@link BudgetExceededException} <em>without</em> calling the model. Loop caps
 * ({@code MAX_STEPS} and friends) bound the number of calls; a budget bounds their cost, which matters when prompts grow
 * every turn.
 * <p>
 * <b>Privacy.</b> Records and log lines hold sizes and timings only, never prompt or reply text, because prompts
 * often contain user data. Tokens are <em>estimated</em> as characters ÷ 4, a common rule of thumb; with a
 * LangChain4j {@code ChatModel} you can read exact counts from {@code ChatResponse.tokenUsage()}, or attach a
 * {@code ChatModelListener} to the model builder.
 * <p>
 * Thread-safe: calls may come from several threads at once (as in the parallel samples). The record history is capped
 * at {@link #MAX_RECORDS}; the totals cover every call.
 */
public final class ObservedLlm implements Function<String, String> {

    static final int MAX_RECORDS = 1_000;

    private static final Logger log = LoggerFactory.getLogger(ObservedLlm.class);

    /** One call: its sequence number, how long it took, sizes, estimated tokens, and whether it failed. */
    public record CallRecord(int seq, long latencyMs, int promptChars, int replyChars, int estimatedTokens, boolean failed) {
    }

    /** Thrown instead of calling the model when a call would spend more than the remaining budget. */
    public static final class BudgetExceededException extends RuntimeException {
        BudgetExceededException(String message) {
            super(message);
        }
    }

    private final Function<String, String> delegate;
    private final int budgetTokens;
    private final LongSupplier nanoClock;
    private final Deque<CallRecord> records = new ArrayDeque<>();
    private int calls;
    private int usedTokens;
    private long totalLatencyMs;

    /**
     * @param budgetTokens the most tokens (prompt + reply, estimated) the wrapped model may use; {@code Integer.MAX_VALUE}
     *                     for no budget
     */
    public ObservedLlm(Function<String, String> delegate, int budgetTokens) {
        this(delegate, budgetTokens, System::nanoTime);
    }

    ObservedLlm(Function<String, String> delegate, int budgetTokens, LongSupplier nanoClock) {
        this.delegate = delegate;
        this.budgetTokens = budgetTokens;
        this.nanoClock = nanoClock;
    }

    /**
     * Calls the wrapped model, measuring it and charging the budget.
     *
     * @throws BudgetExceededException if the prompt alone would exceed the remaining budget (the model is not called)
     */
    @Override
    public String apply(String prompt) {
        int promptTokens = estimateTokens(prompt);
        int seq;
        synchronized (this) {
            if (usedTokens + promptTokens > budgetTokens) {
                throw new BudgetExceededException("token budget of %d would be exceeded (used %d, this prompt ~%d)"
                        .formatted(budgetTokens, usedTokens, promptTokens));
            }
            usedTokens += promptTokens;   // reserve now, so concurrent calls can't all pass the check together
            seq = ++calls;
        }
        long start = nanoClock.getAsLong();
        String reply = null;
        try {
            reply = delegate.apply(prompt);
            return reply;
        } finally {
            long latencyMs = (nanoClock.getAsLong() - start) / 1_000_000;
            int replyChars = reply == null ? 0 : reply.length();
            int replyTokens = reply == null ? 0 : estimateTokens(reply);
            var record = new CallRecord(seq, latencyMs, prompt.length(), replyChars, promptTokens + replyTokens, reply == null);
            synchronized (this) {
                usedTokens += replyTokens;
                totalLatencyMs += latencyMs;
                records.addLast(record);
                if (records.size() > MAX_RECORDS) {
                    records.removeFirst();
                }
            }
            log.debug("llm call #{}: {} ms, prompt {} chars, reply {} chars, ~{} tokens{}", seq, latencyMs,
                    prompt.length(), replyChars, record.estimatedTokens(), record.failed() ? ", FAILED" : "");
        }
    }

    /** Characters ÷ 4, rounded up: a rough but common estimate of tokens for English text. */
    static int estimateTokens(String text) {
        return (text.length() + 3) / 4;
    }

    public synchronized List<CallRecord> records() {
        return List.copyOf(records);
    }

    public synchronized int calls() {
        return calls;
    }

    public synchronized int usedTokens() {
        return usedTokens;
    }

    public synchronized long totalLatencyMs() {
        return totalLatencyMs;
    }

    public int budgetTokens() {
        return budgetTokens;
    }
}
