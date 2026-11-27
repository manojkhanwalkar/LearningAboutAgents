package com.learning.agents.memory;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.CosineSimilarity;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

/**
 * Long-term memory as a small JSON file of facts that outlives the process, plus embedding search over it.
 * The file holds only the fact texts; embeddings are recomputed (in one batch) on load, so changing the embedding
 * model never leaves stale vectors behind. Facts loaded from the file go through the same rules as {@link #add}:
 * normalised, de-duplicated, at most {@link #MAX_FACT_CHARS} long, and only the newest {@link #MAX_FACTS} kept.
 * Writes go to a temporary file that is then moved into place, so a crash mid-write never leaves a half-written
 * memory. There is no expiry: a fact stays until it is forgotten or pushed out by newer ones.
 * Not thread-safe: use one instance per user, from one thread.
 */
public final class FactStore {

    static final int MAX_FACTS = 200;
    static final int MAX_FACT_CHARS = 200;

    private static final ObjectMapper JSON = new ObjectMapper();

    private record Entry(String text, Embedding embedding) {
    }

    private record Scored(String text, double score) {
    }

    private final Path file;
    private final EmbeddingModel embeddings;
    private final List<Entry> entries = new ArrayList<>();

    /**
     * Opens the store, loading any facts already saved in {@code file}.
     *
     * @throws UncheckedIOException if the file exists but cannot be read, is not a JSON list of strings, or holds nulls
     */
    public FactStore(Path file, EmbeddingModel embeddings) {
        this.file = file;
        this.embeddings = embeddings;
        if (!Files.exists(file)) {
            return;
        }
        List<String> saved;
        try {
            saved = JSON.readValue(file.toFile(), new TypeReference<>() {
            });
            if (saved == null || saved.stream().anyMatch(Objects::isNull)) {
                throw new IOException("memory file must be a JSON list of strings");
            }
        } catch (IOException e) {
            throw new UncheckedIOException("Cannot read memory file " + file, e);
        }
        List<String> kept = new ArrayList<>();
        for (String fact : saved) {
            String text = normalize(fact);
            if (isAcceptable(text) && kept.stream().noneMatch(k -> sameFact(k, text))) {
                kept.add(text);
            }
        }
        // Keep the newest facts, as add() does, and embed them in one batch rather than one call per fact.
        List<String> newest = kept.subList(Math.max(0, kept.size() - MAX_FACTS), kept.size());
        if (!newest.isEmpty()) {
            List<Embedding> vectors = embeddings.embedAll(newest.stream().map(TextSegment::from).toList()).content();
            for (int i = 0; i < newest.size(); i++) {
                entries.add(new Entry(newest.get(i), vectors.get(i)));
            }
        }
    }

    /**
     * Adds a fact unless it is blank, longer than {@link #MAX_FACT_CHARS}, or already stored (ignoring case and
     * spacing); drops the oldest fact beyond {@link #MAX_FACTS} and saves the file.
     *
     * @return whether the fact was added
     */
    public boolean add(String fact) {
        String text = normalize(fact);
        if (!isAcceptable(text) || entries.stream().anyMatch(e -> sameFact(e.text(), text))) {
            return false;
        }
        entries.add(new Entry(text, embed(text)));
        while (entries.size() > MAX_FACTS) {
            entries.removeFirst();
        }
        save();
        return true;
    }

    /**
     * Removes every fact containing {@code phrase} (case-insensitive) and saves. A blank phrase removes nothing, so a
     * bad caller can never wipe the whole memory by accident.
     *
     * @return how many facts were removed
     */
    public int forget(String phrase) {
        if (phrase == null || phrase.isBlank()) {
            return 0;
        }
        String needle = phrase.strip().toLowerCase(Locale.ROOT);
        int before = entries.size();
        entries.removeIf(e -> e.text().toLowerCase(Locale.ROOT).contains(needle));
        int removed = before - entries.size();
        if (removed > 0) {
            save();
        }
        return removed;
    }

    /**
     * Returns up to {@code k} facts whose cosine similarity to {@code query} is at least {@code minScore}, best
     * first. This is raw cosine (-1 to 1, 0 = unrelated), not the (cosine + 1) / 2 relevance score of LangChain4j
     * stores. A blank query recalls nothing.
     */
    public List<String> recall(String query, int k, double minScore) {
        if (query == null || query.isBlank()) {
            return List.of();
        }
        Embedding q = embed(query);
        return entries.stream()
                .map(e -> new Scored(e.text(), CosineSimilarity.between(q, e.embedding())))
                .filter(scored -> scored.score() >= minScore)
                .sorted(Comparator.comparingDouble(Scored::score).reversed())
                .limit(k)
                .map(Scored::text)
                .toList();
    }

    public List<String> all() {
        return entries.stream().map(Entry::text).toList();
    }

    private static String normalize(String fact) {
        return fact.strip().replaceAll("\\s+", " ");
    }

    private static boolean isAcceptable(String text) {
        return !text.isEmpty() && text.length() <= MAX_FACT_CHARS;
    }

    private static boolean sameFact(String a, String b) {
        return a.toLowerCase(Locale.ROOT).equals(b.toLowerCase(Locale.ROOT));
    }

    private Embedding embed(String text) {
        return embeddings.embed(TextSegment.from(text)).content();
    }

    private void save() {
        Path tmp = null;
        try {
            Path parent = file.toAbsolutePath().getParent();
            Files.createDirectories(parent);
            tmp = Files.createTempFile(parent, "memory", ".tmp");
            JSON.writeValue(tmp.toFile(), all());
            Files.move(tmp, file, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } catch (IOException e) {
            deleteQuietly(tmp);
            throw new UncheckedIOException("Cannot write memory file " + file, e);
        }
    }

    private static void deleteQuietly(Path tmp) {
        if (tmp == null) {
            return;
        }
        try {
            Files.deleteIfExists(tmp);
        } catch (IOException ignored) {
            // Best effort: the write already failed, and that is the error the caller needs to see.
        }
    }
}
