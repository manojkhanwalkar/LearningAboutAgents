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
 * <b>Budget.</b> Before each call it reserves the prompt's estimated tokens <em>plus</em> {@code maxReplyTokens}, the
 * most the reply may use; if that doesn't fit in what is left, the call is refused with {@link BudgetExceededException}
 * without calling the model. After the call the reservation is settled to the reply's real size. So the budget holds
 * as long as replies stay within {@code maxReplyTokens}: set the model's own output limit (Ollama {@code numPredict},
 * LangChain4j {@code maxTokens}) to the same value. A reply that still runs over is charged in full and flagged in its
 * record. Loop caps ({@code MAX_STEPS} and friends) bound the number of calls; a budget bounds their cost.
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
    public record CallRecord(int seq, long latencyMs, int promptChars, int replyChars, int estimatedTokens, boolean failed,
                             boolean replyOverran) {
    }

    /** Thrown instead of calling the model when a call would spend more than the remaining budget. */
    public static final class BudgetExceededException extends RuntimeException {
        BudgetExceededException(String message) {
            super(message);
        }
    }

    static final int DEFAULT_MAX_REPLY_TOKENS = 1_024;

    private final Function<String, String> delegate;
    private final int budgetTokens;
    private final int maxReplyTokens;
    private final LongSupplier nanoClock;
    private final Deque<CallRecord> records = new ArrayDeque<>();
    private int calls;
    private int usedTokens;
    private long totalLatencyMs;

    /**
     * Observes {@code delegate} with a budget, assuming replies of at most {@value #DEFAULT_MAX_REPLY_TOKENS} tokens.
     *
     * @param budgetTokens the most tokens (prompt + reply, estimated) the wrapped model may use; {@code Integer.MAX_VALUE}
     *                     for no budget
     */
    public ObservedLlm(Function<String, String> delegate, int budgetTokens) {
        this(delegate, budgetTokens, DEFAULT_MAX_REPLY_TOKENS);
    }

    /** Same, with the most tokens a reply may use; give the model the same output limit so the budget holds. */
    public ObservedLlm(Function<String, String> delegate, int budgetTokens, int maxReplyTokens) {
        this(delegate, budgetTokens, maxReplyTokens, System::nanoTime);
    }

    ObservedLlm(Function<String, String> delegate, int budgetTokens, int maxReplyTokens, LongSupplier nanoClock) {
        this.delegate = delegate;
        this.budgetTokens = budgetTokens;
        this.maxReplyTokens = maxReplyTokens;
        this.nanoClock = nanoClock;
    }

    /**
     * Calls the wrapped model, measuring it and charging the budget.
     *
     * @throws BudgetExceededException if the prompt plus the largest allowed reply would exceed the remaining budget
     *                                  (the model is not called)
     */
    @Override
    public String apply(String prompt) {
        int promptTokens = estimateTokens(prompt);
        long reservation = (long) promptTokens + maxReplyTokens;
        int seq;
        synchronized (this) {
            if (usedTokens + reservation > budgetTokens) {
                throw new BudgetExceededException("token budget of %d would be exceeded (used %d, this prompt ~%d + up to %d for the reply)"
                        .formatted(budgetTokens, usedTokens, promptTokens, maxReplyTokens));
            }
            usedTokens += (int) reservation;   // reserve now, so concurrent calls can't all pass the check together
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
            boolean overran = replyTokens > maxReplyTokens;
            var record = new CallRecord(seq, latencyMs, prompt.length(), replyChars, promptTokens + replyTokens, reply == null, overran);
            synchronized (this) {
                usedTokens += replyTokens - maxReplyTokens;   // settle the reservation to the real reply size
                totalLatencyMs += latencyMs;
                records.addLast(record);
                if (records.size() > MAX_RECORDS) {
                    records.removeFirst();
                }
            }
            log.debug("llm call #{}: {} ms, prompt {} chars, reply {} chars, ~{} tokens{}{}", seq, latencyMs,
                    prompt.length(), replyChars, record.estimatedTokens(), record.failed() ? ", FAILED" : "",
                    overran ? ", REPLY OVER LIMIT" : "");
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

    public int maxReplyTokens() {
        return maxReplyTokens;
    }

    public int budgetTokens() {
        return budgetTokens;
    }
}
