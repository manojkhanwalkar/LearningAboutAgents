package com.learning.agents.rag;

import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * A tiny in-memory retriever that ranks documents by TF-IDF cosine similarity to the query. It stands in for an
 * embedding model and vector store so the RAG sample has no extra dependencies. It finds documents that share
 * <em>words</em> with the query, while embeddings also find documents that share <em>meaning</em>. To use embeddings
 * later, swap this for LangChain4j's {@code OllamaEmbeddingModel} with an {@code InMemoryEmbeddingStore}.
 * The index is built once in the constructor and is read-only afterwards, so searches are thread-safe.
 */
public final class KeywordRetriever {

    /** A source document. {@code id} is what the answer cites. */
    public record Doc(String id, String text) {
    }

    /** A search hit with its similarity score in (0, 1]. */
    public record Hit(Doc doc, double score) {
    }

    private static final Set<String> STOP_WORDS = Set.of(
            "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "how", "i", "in", "is", "it",
            "my", "of", "on", "or", "the", "to", "what", "when", "which", "with", "you", "your");

    private final List<Doc> docs;
    private final Map<String, Double> idf;
    private final List<Map<String, Double>> vectors;

    public KeywordRetriever(List<Doc> docs) {
        this.docs = List.copyOf(docs);
        List<Set<String>> termSets = this.docs.stream().map(d -> Set.copyOf(tokenize(d.text()))).toList();
        Map<String, Long> docFrequency = termSets.stream().flatMap(Set::stream)
                .collect(Collectors.groupingBy(Function.identity(), Collectors.counting()));
        int n = this.docs.size();
        // Smoothed IDF: words found in every document still count a little; rare words count most.
        this.idf = docFrequency.entrySet().stream()
                .collect(Collectors.toUnmodifiableMap(Map.Entry::getKey, e -> Math.log(1.0 + n / (double) e.getValue())));
        this.vectors = this.docs.stream().map(d -> vector(tokenize(d.text()))).toList();
    }

    /** Returns up to {@code k} documents with a positive score, best first. */
    public List<Hit> search(String query, int k) {
        Map<String, Double> q = vector(tokenize(query));
        return IntStream.range(0, docs.size())
                .mapToObj(i -> new Hit(docs.get(i), cosine(q, vectors.get(i))))
                .filter(hit -> hit.score() > 0)
                .sorted(Comparator.comparingDouble(Hit::score).reversed())
                .limit(k)
                .toList();
    }

    static List<String> tokenize(String text) {
        return Arrays.stream(text.toLowerCase(Locale.ROOT).split("[^\\p{L}\\p{N}]+"))
                .filter(token -> token.length() > 1 && !STOP_WORDS.contains(token))
                .toList();
    }

    /** Term frequency times IDF. Words never seen while indexing get no weight, because no document can match them. */
    private Map<String, Double> vector(List<String> tokens) {
        Map<String, Double> v = new HashMap<>();
        for (String token : tokens) {
            Double weight = idf.get(token);
            if (weight != null) {
                v.merge(token, weight, Double::sum);
            }
        }
        return v;
    }

    private static double cosine(Map<String, Double> a, Map<String, Double> b) {
        if (a.isEmpty() || b.isEmpty()) {
            return 0;
        }
        double dot = a.entrySet().stream().mapToDouble(e -> e.getValue() * b.getOrDefault(e.getKey(), 0.0)).sum();
        return dot / (norm(a) * norm(b));
    }

    private static double norm(Map<String, Double> v) {
        return Math.sqrt(v.values().stream().mapToDouble(x -> x * x).sum());
    }
}
