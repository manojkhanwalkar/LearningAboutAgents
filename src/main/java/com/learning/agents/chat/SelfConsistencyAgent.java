package com.learning.agents.chat;

import com.learning.agents.llm.LlmProvider;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ExecutionException;
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
 * {@code OllamaConfig} default of 0.2.
 */
public class SelfConsistencyAgent {

    static final int SAMPLES = 5;

    private static final Pattern ANSWER = Pattern.compile("ANSWER:\\s*(.+)", Pattern.CASE_INSENSITIVE);

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
    static Optional<Vote> solve(Function<String, String> llm, String question, int samples)
            throws InterruptedException, ExecutionException {
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
            for (var future : futures) {
                replies.add(future.get());
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
                .map(answer -> answer.strip().replaceAll("[.!]+$", "").toLowerCase(Locale.ROOT))
                .filter(answer -> !answer.isEmpty());
    }

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
