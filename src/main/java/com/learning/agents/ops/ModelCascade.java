package com.learning.agents.ops;

import com.learning.agents.OllamaConfig;
import com.learning.agents.llm.LlmProvider;
import com.learning.agents.llm.Replies;
import com.learning.agents.llm.ScriptedLlm;
import dev.langchain4j.exception.RetriableException;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.FutureTask;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.random.RandomGenerator;
import java.util.regex.Pattern;

/**
 * A model cascade with resilience, the two things that make local models practical in production.
 * <p>
 * <b>Cascade.</b> Ask a small, fast model first and have it rate its own confidence. Only when the confidence is
 * below {@link #MIN_CONFIDENCE}, or the reply can't be read, escalate to a larger, slower model. Most questions are
 * easy, so most never pay for the big model.
 * <p>
 * <b>Resilience.</b> Every call gets a deadline ({@link #CALL_TIMEOUT} by default). A failure is retried only if
 * it is <em>recoverable</em>: a timeout, a LangChain4j {@link RetriableException} (server error, rate limit) or a
 * {@link TransientModelException}. Retries wait with exponential backoff and <em>full jitter</em>: a random wait
 * between 0 and min({@link #MAX_DELAY}, {@link #BASE_DELAY} × 2^(attempt−1)), so many clients don't retry in lock-step.
 * A bad request is never retried. If the large model still fails after {@link #MAX_ATTEMPTS}, a fallback model
 * answers. There is exactly one retry layer: the Ollama models are built with {@code maxRetries(0)}.
 *
 * <pre>
 *   small (retry) ─┬─ confident → answer
 *                  └─ unsure / failed → large (retry) ─┬─ ok → answer
 *                                                      └─ failed → fallback (retry) → answer, or give up
 * </pre>
 */
public class ModelCascade {

    static final int MIN_CONFIDENCE = 70;
    static final int MAX_ATTEMPTS = 3;
    static final Duration BASE_DELAY = Duration.ofMillis(200);
    static final Duration MAX_DELAY = Duration.ofSeconds(2);
    static final Duration CALL_TIMEOUT = Duration.ofSeconds(60);

    private static final Pattern LEADING_NUMBER = Pattern.compile("^(\\d{1,3})");

    /** A thread-safe random source: each call draws from the calling thread's own generator. */
    static final RandomGenerator THREAD_SAFE_RANDOM = new RandomGenerator() {
        @Override
        public long nextLong() {
            return ThreadLocalRandom.current().nextLong();
        }
    };

    /** A failure worth retrying, such as an overloaded server. Real clients throw their own retriable exceptions. */
    static final class TransientModelException extends RuntimeException {
        TransientModelException(String message) {
            super(message);
        }
    }

    /** How the wait between retries is spent; replaced in tests so they don't actually sleep. */
    @FunctionalInterface
    interface Sleeper {
        void sleep(Duration duration) throws InterruptedException;
    }

    /** The answer (empty if every model failed), which model gave it, and a step-by-step trace. */
    record Result(Optional<String> answer, String answeredBy, List<String> trace) {
    }

    private final Function<String, String> small;
    private final Function<String, String> large;
    private final Function<String, String> fallback;
    private final Duration callTimeout;
    private final Sleeper sleeper;
    private final RandomGenerator random;

    ModelCascade(Function<String, String> small, Function<String, String> large, Function<String, String> fallback,
                 Duration callTimeout, Sleeper sleeper, RandomGenerator random) {
        this.small = small;
        this.large = large;
        this.fallback = fallback;
        this.callTimeout = callTimeout;
        this.sleeper = sleeper;
        this.random = random;
    }

    /** Runs the cascade for one question. Never throws for model failures; they show up in the trace. */
    public Result ask(String question) throws InterruptedException {
        List<String> trace = new ArrayList<>();
        Optional<String> smallReply = callWithRetry("small", small, """
                Answer the question in one sentence, then rate how sure you are from 0 to 100.
                Reply as:
                ANSWER: <answer>
                CONFIDENCE: <0-100>
                Question: %s""".formatted(question), trace);
        if (smallReply.isPresent()) {
            Optional<Integer> confidence = confidence(smallReply.get());
            Optional<String> answer = answer(smallReply.get());
            if (answer.isPresent() && confidence.orElse(0) >= MIN_CONFIDENCE) {
                trace.add(String.format(Locale.ROOT, "small: confident (%d), done", confidence.get()));
                return new Result(answer, "small", trace);
            }
            trace.add("small: not confident enough (%s), escalating".formatted(confidence.map(String::valueOf).orElse("unreadable")));
        }
        String bigPrompt = "Answer the question carefully in one or two sentences.\nQuestion: " + question;
        Optional<String> largeReply = callWithRetry("large", large, bigPrompt, trace);
        if (largeReply.isPresent()) {
            return new Result(Optional.of(largeReply.get().strip()), "large", trace);
        }
        Optional<String> fallbackReply = callWithRetry("fallback", fallback, bigPrompt, trace);
        return fallbackReply
                .map(reply -> new Result(Optional.of(reply.strip()), "fallback", trace))
                .orElseGet(() -> new Result(Optional.empty(), "none", trace));
    }

    /**
     * Calls one model with a deadline, retrying recoverable failures with full-jitter backoff.
     *
     * @return the reply, or empty if the model failed permanently or ran out of attempts
     */
    Optional<String> callWithRetry(String name, Function<String, String> model, String prompt, List<String> trace)
            throws InterruptedException {
        for (int attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
            try {
                return Optional.of(callWithTimeout(model, prompt));
            } catch (RuntimeException | TimeoutException e) {
                if (!isRetryable(e)) {
                    trace.add("%s: failed permanently (%s), not retrying".formatted(name, describe(e)));
                    return Optional.empty();
                }
                if (attempt == MAX_ATTEMPTS) {
                    trace.add("%s: failed (%s), out of attempts".formatted(name, describe(e)));
                    return Optional.empty();
                }
                Duration wait = backoff(attempt);
                trace.add(String.format(Locale.ROOT, "%s: attempt %d failed (%s), retrying in %d ms", name, attempt, describe(e), wait.toMillis()));
                sleeper.sleep(wait);
            }
        }
        return Optional.empty();   // not reached: the loop always returns
    }

    /**
     * Runs the call on its own virtual thread and stops <em>waiting</em> at the deadline, interrupting the call. A call
     * that ignores interrupts may keep running in the background, but the caller is never held past the deadline.
     * (Closing an executor in try-with-resources would wait for that call to finish, which defeats the deadline.)
     */
    private String callWithTimeout(Function<String, String> model, String prompt) throws TimeoutException, InterruptedException {
        var task = new FutureTask<>(() -> model.apply(prompt));
        Thread.ofVirtual().name("model-call").start(task);
        try {
            return task.get(callTimeout.toMillis(), TimeUnit.MILLISECONDS);
        } catch (TimeoutException | InterruptedException e) {
            task.cancel(true);   // on our own interrupt too, so Ctrl-C doesn't wait for the model
            throw e;
        } catch (ExecutionException e) {
            Throwable cause = e.getCause();
            if (cause instanceof Error error) {
                throw error;   // not a model failure: never hide it behind a fallback
            }
            throw cause instanceof RuntimeException re ? re : new IllegalStateException(cause);
        }
    }

    /** Full jitter: a uniformly random wait between 0 and min(MAX_DELAY, BASE_DELAY × 2^(attempt-1)). */
    Duration backoff(int attempt) {
        long cap = Math.min(MAX_DELAY.toMillis(), BASE_DELAY.toMillis() << Math.min(attempt - 1, 20));
        return Duration.ofMillis(random.nextLong(cap + 1));
    }

    /** Timeouts and server-side or rate-limit failures are worth retrying; bad requests and bugs are not. */
    static boolean isRetryable(Throwable e) {
        for (Throwable t = e; t != null; t = t.getCause()) {
            if (t instanceof TimeoutException || t instanceof RetriableException || t instanceof TransientModelException) {
                return true;
            }
        }
        return false;
    }

    /** The number on the {@code CONFIDENCE:} line, capped at 100; decorated labels such as {@code **CONFIDENCE:**} work. */
    static Optional<Integer> confidence(String reply) {
        return Replies.lineAfter(reply, "CONFIDENCE:")
                .map(LEADING_NUMBER::matcher)
                .filter(java.util.regex.Matcher::find)
                .map(m -> Math.min(100, Integer.parseInt(m.group(1))));
    }

    /** The text on the {@code ANSWER:} line (the confidence goes on its own line, so one line is the answer). */
    static Optional<String> answer(String reply) {
        return Replies.lineAfter(reply, "ANSWER:");
    }

    private static String describe(Throwable e) {
        return e instanceof TimeoutException ? "timed out" : e.getClass().getSimpleName() + ": " + e.getMessage();
    }

    /** The small placeholder model: confident on easy questions, unsure on the hard one. */
    static ScriptedLlm smallPlaceholder() {
        return ScriptedLlm.create()
                .when("capital of France", "ANSWER: Paris.\nCONFIDENCE: 98")
                .when("Riemann", "ANSWER: I think it's about prime numbers?\nCONFIDENCE: 30")
                .otherwise("ANSWER: Not sure.\nCONFIDENCE: 10");
    }

    /** The large placeholder model: overloaded on its first call (a transient error), then fine, except for "Lobachevsky". */
    static Function<String, String> largePlaceholder() {
        var calls = new AtomicInteger();
        ScriptedLlm replies = ScriptedLlm.create()
                .when("Riemann", "It conjectures that every non-trivial zero of the zeta function has real part 1/2.");
        return prompt -> {
            if (calls.getAndIncrement() == 0 || prompt.contains("Lobachevsky")) {
                throw new TransientModelException("model server overloaded (503)");
            }
            return replies.apply(prompt);
        };
    }

    static ScriptedLlm fallbackPlaceholder() {
        return ScriptedLlm.create().otherwise("Lobachevsky founded hyperbolic (non-Euclidean) geometry.");
    }

    public static void main(String[] args) throws Exception {
        String smallName = System.getenv().getOrDefault("CASCADE_SMALL_MODEL", OllamaConfig.modelName());
        String largeName = System.getenv().getOrDefault("CASCADE_LARGE_MODEL", OllamaConfig.modelName());
        String fallbackName = System.getenv().getOrDefault("CASCADE_FALLBACK_MODEL", OllamaConfig.modelName());
        var cascade = new ModelCascade(
                LlmProvider.resolve(smallPlaceholder(), () -> OllamaConfig.chatModel(smallName, 0)::chat),
                LlmProvider.resolve(largePlaceholder(), () -> OllamaConfig.chatModel(largeName, 0)::chat),
                LlmProvider.resolve(fallbackPlaceholder(), () -> OllamaConfig.chatModel(fallbackName, 0)::chat),
                CALL_TIMEOUT, Thread::sleep, THREAD_SAFE_RANDOM);

        var questions = args.length > 0
                ? List.of(String.join(" ", args))
                : List.of("What is the capital of France?",
                "In one sentence, what does the Riemann hypothesis say?",
                "What is Lobachevsky known for?");
        for (String question : questions) {
            Result result = cascade.ask(question);
            System.out.println("Q: " + question);
            result.trace().forEach(step -> System.out.println("  " + step));
            System.out.println("A (" + result.answeredBy() + "): " + result.answer().orElse("(no model could answer)") + "\n");
        }
    }
}
