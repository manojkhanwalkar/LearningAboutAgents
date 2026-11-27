package com.learning.agents.ops;

import com.learning.agents.llm.HashingEmbeddingModel;
import com.learning.agents.llm.LlmProvider;
import com.learning.agents.llm.ScriptedLlm;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.CosineSimilarity;

import java.time.Clock;
import java.time.Duration;
import java.time.Instant;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Optional;
import java.util.function.Function;

/**
 * A semantic cache: reuse an earlier answer when a new question <em>means</em> the same as an old one ("what's the
 * capital of France" after "What is the capital of France?"), judged by embedding similarity. Exact-match caches miss
 * these; a semantic cache saves a model call each time.
 * <p>
 * The danger is the false hit: "capital of Spain" is also close to "capital of France", and returning Paris for it is
 * worse than no cache at all. So the similarity threshold is the most important setting, and it must be tuned for
 * the embedding model; {@link #main} runs the same questions at a safe and an unsafe threshold. Use a semantic cache for
 * answers that don't depend on details a near-duplicate might change (numbers, names, dates), or not at all.
 * <p>
 * Bounded in two ways: at most {@link #MAX_ENTRIES} entries (least recently used evicted first) and a time-to-live,
 * so stale answers expire. Thread-safe; the model call itself happens outside the lock.
 */
public final class SemanticCache implements Function<String, String> {

    static final int MAX_ENTRIES = 100;

    private record Entry(String question, Embedding embedding, String answer, Instant storedAt) {
    }

    /** A lookup result, for display: the answer, whether it came from the cache, and the matched question. */
    public record Lookup(String answer, boolean hit, Optional<String> matched, double similarity) {
    }

    private final Function<String, String> llm;
    private final EmbeddingModel embeddings;
    private final double threshold;
    private final Duration ttl;
    private final Clock clock;
    private final LinkedHashMap<String, Entry> entries = new LinkedHashMap<>(16, 0.75f, true);   // access order = LRU
    private int hits;
    private int misses;

    /**
     * @param threshold minimum cosine similarity (−1 to 1) for a cached answer to be reused
     * @param ttl       how long an answer stays valid
     */
    public SemanticCache(Function<String, String> llm, EmbeddingModel embeddings, double threshold, Duration ttl, Clock clock) {
        this.llm = llm;
        this.embeddings = embeddings;
        this.threshold = threshold;
        this.ttl = ttl;
        this.clock = clock;
    }

    @Override
    public String apply(String question) {
        return lookup(question).answer();
    }

    /** Returns a cached answer if a similar enough, unexpired question was asked before; otherwise asks the model. */
    public Lookup lookup(String question) {
        Embedding q = embeddings.embed(question).content();
        Instant now = clock.instant();
        synchronized (this) {
            Entry best = null;
            double bestScore = -1;
            for (Iterator<Entry> it = entries.values().iterator(); it.hasNext(); ) {
                Entry entry = it.next();
                if (entry.storedAt().plus(ttl).isBefore(now)) {
                    it.remove();   // expired
                    continue;
                }
                double score = CosineSimilarity.between(q, entry.embedding());
                if (score > bestScore) {
                    best = entry;
                    bestScore = score;
                }
            }
            if (best != null && bestScore >= threshold) {
                entries.get(best.question());   // touch, for LRU order
                hits++;
                return new Lookup(best.answer(), true, Optional.of(best.question()), bestScore);
            }
            misses++;
        }
        String answer = llm.apply(question);   // outside the lock: a slow model call must not block other lookups
        synchronized (this) {
            entries.put(question, new Entry(question, q, answer, now));
            if (entries.size() > MAX_ENTRIES) {
                entries.remove(entries.keySet().iterator().next());   // least recently used
            }
        }
        return new Lookup(answer, false, Optional.empty(), 0);
    }

    public synchronized int hits() {
        return hits;
    }

    public synchronized int misses() {
        return misses;
    }

    public synchronized int size() {
        return entries.size();
    }

    static ScriptedLlm placeholder() {
        return ScriptedLlm.create()
                .when("France", "Paris")
                .when("Spain", "Madrid")
                .otherwise("I don't know.");
    }

    static final List<String> QUESTIONS = List.of(
            "What is the capital of France?",
            "what's the capital of France",
            "What is the capital of Spain?");

    static void run(String title, SemanticCache cache) {
        System.out.println(title);
        for (String question : QUESTIONS) {
            Lookup lookup = cache.lookup(question);
            System.out.printf("  %-32s → %-7s %s%n", question, lookup.answer(), lookup.hit()
                    ? "(cache hit, %.2f similar to \"%s\")".formatted(lookup.similarity(), lookup.matched().orElse(""))
                    : "(model call)");
        }
        System.out.printf("  hits %d, misses %d%n%n", cache.hits(), cache.misses());
    }

    public static void main(String[] args) {
        Function<String, String> llm = LlmProvider.resolve(placeholder());
        EmbeddingModel embeddings = LlmProvider.resolveEmbeddings(new HashingEmbeddingModel());
        run("Threshold 0.90 (safe for these embeddings):",
                new SemanticCache(llm, embeddings, 0.90, Duration.ofHours(1), Clock.systemUTC()));
        run("Threshold 0.40 (too loose: watch Spain get France's answer):",
                new SemanticCache(llm, embeddings, 0.40, Duration.ofHours(1), Clock.systemUTC()));
    }
}
