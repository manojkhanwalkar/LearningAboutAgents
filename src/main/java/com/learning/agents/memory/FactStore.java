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

/**
 * Long-term memory as a small JSON file of facts that outlives the process, plus embedding search over it.
 * The file holds only the fact texts; embeddings are recomputed on load, so changing the embedding model never
 * leaves stale vectors behind. Writes go to a temporary file that is then moved into place, so a crash mid-write
 * never leaves a half-written memory. At most {@link #MAX_FACTS} facts are kept; the oldest are dropped first.
 * Not thread-safe: use one instance per user, from one thread.
 */
public final class FactStore {

    static final int MAX_FACTS = 200;

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
     * @throws UncheckedIOException if the file exists but cannot be read or parsed
     */
    public FactStore(Path file, EmbeddingModel embeddings) {
        this.file = file;
        this.embeddings = embeddings;
        if (Files.exists(file)) {
            try {
                List<String> saved = JSON.readValue(file.toFile(), new TypeReference<>() {
                });
                saved.stream().limit(MAX_FACTS).forEach(text -> entries.add(new Entry(text, embed(text))));
            } catch (IOException e) {
                throw new UncheckedIOException("Cannot read memory file " + file, e);
            }
        }
    }

    /** Adds a fact unless an identical one (ignoring case and spacing) is already stored; saves the file. */
    public boolean add(String fact) {
        String text = fact.strip().replaceAll("\\s+", " ");
        String key = text.toLowerCase(Locale.ROOT);
        if (text.isEmpty() || entries.stream().anyMatch(e -> e.text().toLowerCase(Locale.ROOT).equals(key))) {
            return false;
        }
        entries.add(new Entry(text, embed(text)));
        while (entries.size() > MAX_FACTS) {
            entries.removeFirst();
        }
        save();
        return true;
    }

    /** Removes every fact containing {@code phrase} (case-insensitive) and saves; returns how many were removed. */
    public int forget(String phrase) {
        String needle = phrase.toLowerCase(Locale.ROOT);
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
     * first. This is raw cosine (-1 to 1, 0 = unrelated), not the (cosine + 1) / 2 relevance score of LangChain4j stores.
     */
    public List<String> recall(String query, int k, double minScore) {
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

    private Embedding embed(String text) {
        return embeddings.embed(TextSegment.from(text)).content();
    }

    private void save() {
        try {
            Path parent = file.toAbsolutePath().getParent();
            Files.createDirectories(parent);
            Path tmp = Files.createTempFile(parent, "memory", ".tmp");
            JSON.writeValue(tmp.toFile(), all());
            Files.move(tmp, file, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } catch (IOException e) {
            throw new UncheckedIOException("Cannot write memory file " + file, e);
        }
    }
}
