package com.learning.agents.chat;

import com.learning.agents.llm.LlmProvider;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Self-consistency (Wang et al. 2022) plus parallel calls: ask the same question {@link #SAMPLES} times
 * concurrently, pull the final answer out of each reply, and take a majority vote. One sample can reason its way to
 * a wrong answer; several independent samples rarely all make the same mistake. The agreement ratio is a cheap
 * confidence signal: low agreement means "don't trust this answer".
 * <p>
 * Votes only differ if the model samples with some randomness, so with a real model raise the temperature above the
 * {@code OllamaConfig} default of 0.2. A sample that throws or misses the {@link #SAMPLE_TIMEOUT} deadline abstains.
 */
public class SelfConsistencyAgent {

    static final int SAMPLES = 5;

    /** How long to wait for all samples together; one that fails or is still running then abstains instead of sinking the vote. */
    static final Duration SAMPLE_TIMEOUT = Duration.ofMinutes(3);

    private static final Logger log = LoggerFactory.getLogger(SelfConsistencyAgent.class);
    private static final Pattern ANSWER = Pattern.compile("ANSWER:[*_]*[ \\t]*(.+)", Pattern.CASE_INSENSITIVE);

    /** The winning answer, how many samples gave it, and how many samples produced a usable answer. */
    record Vote(String answer, long votes, int valid) {
        double agreement() {
            return valid == 0 ? 0 : (double) votes / valid;
        }
    }

    /**
     * Asks {@code question} {@code samples} times in parallel and returns the majority answer.
     *
     * @param llm maps a prompt to a model reply; called from several threads at once, so it must be thread-safe
     * @return the vote, or empty if no sample produced an {@code ANSWER:} line
     */
    static Optional<Vote> solve(Function<String, String> llm, String question, int samples) throws InterruptedException {
        return solve(llm, question, samples, SAMPLE_TIMEOUT);
    }

    /**
     * Same as {@link #solve(Function, String, int)} with the deadline passed in, so it can be tested.
     *
     * @param timeout how long to wait for all samples together; unfinished samples are cancelled and abstain
     */
    static Optional<Vote> solve(Function<String, String> llm, String question, int samples, Duration timeout)
            throws InterruptedException {
        String prompt = """
                Think step by step, then give the final answer on its own last line as "ANSWER: <answer>".
                Question: %s""".formatted(question);

        // A virtual thread per call: the calls spend their time waiting on the network, not using the CPU.
        List<String> replies = new ArrayList<>();
        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            List<Future<String>> futures = new ArrayList<>();
            for (int i = 0; i < samples; i++) {
                futures.add(executor.submit(() -> llm.apply(prompt)));
            }
            long deadline = System.nanoTime() + timeout.toNanos();
            for (var future : futures) {
                try {
                    replies.add(future.get(Math.max(0, deadline - System.nanoTime()), TimeUnit.NANOSECONDS));
                } catch (ExecutionException | TimeoutException e) {
                    // One bad sample shouldn't lose the others: it simply doesn't vote.
                    future.cancel(true);
                    log.warn("A sample failed and abstains: {}", e.toString());
                }
            }
        }
        return vote(replies.stream().map(SelfConsistencyAgent::extractAnswer).flatMap(Optional::stream).toList());
    }

    /** Takes the text after the last {@code ANSWER:}, normalised so "42", "42." and " 42 " count as one vote. */
    static Optional<String> extractAnswer(String reply) {
        var matcher = ANSWER.matcher(reply);
        String last = null;
        while (matcher.find()) {
            last = matcher.group(1);
        }
        return Optional.ofNullable(last)
                .map(answer -> answer.replaceAll("[*_`]", "").strip().replaceAll("[.!]+$", "").toLowerCase(Locale.ROOT))
                .filter(answer -> !answer.isEmpty());
    }

    /** Counts identical answers and returns the most common one; empty if {@code answers} is empty. */
    static Optional<Vote> vote(List<String> answers) {
        Map<String, Long> counts = answers.stream().collect(Collectors.groupingBy(Function.identity(), Collectors.counting()));
        return counts.entrySet().stream()
                // Most votes wins; a tie goes to the alphabetically first answer so the result is repeatable.
                .max(Map.Entry.<String, Long>comparingByValue()
                        .thenComparing(Map.Entry.<String, Long>comparingByKey().reversed()))
                .map(winner -> new Vote(winner.getKey(), winner.getValue(), answers.size()));
    }

    /** A placeholder that imitates a noisy model: four samples reason correctly and one slips. */
    static Function<String, String> placeholder() {
        var calls = new AtomicInteger();
        return prompt -> calls.getAndIncrement() % SAMPLES == 2
                ? "16 balls minus 5 is... 12.\nANSWER: 12"
                : "Half of 16 is 8 golf balls, half of those are blue.\nANSWER: 4";
    }

    public static void main(String[] args) throws Exception {
        String question = args.length > 0
                ? String.join(" ", args)
                : "A juggler has 16 balls. Half are golf balls and half of the golf balls are blue. How many blue golf balls?";

        System.out.println("Question: " + question);
        solve(LlmProvider.resolve(placeholder()), question, SAMPLES).ifPresentOrElse(
                vote -> System.out.printf("Answer: %s (%d of %d samples agree, %.0f%%)%n",
                        vote.answer(), vote.votes(), vote.valid(), vote.agreement() * 100),
                () -> System.out.println("No sample produced an ANSWER: line."));
    }
}
