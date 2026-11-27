package com.learning.agents.llm;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;

import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * A placeholder {@link EmbeddingModel} that needs no server: each word is hashed into one of {@link #DIMENSIONS}
 * buckets, and the vector is normalised to length 1. Texts that share words get a high cosine similarity, so the
 * embedding samples behave sensibly without Ollama. Unlike a real embedding model it knows nothing about meaning:
 * "car" and "automobile" are unrelated to it. Set {@code LLM_BACKEND=ollama} to use a real model such as
 * {@code nomic-embed-text}. Stateless and thread-safe.
 */
public final class HashingEmbeddingModel implements EmbeddingModel {

    public static final int DIMENSIONS = 512;

    private static final Set<String> STOP_WORDS = Set.of(
            "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "how", "i", "in", "is",
            "it", "me", "my", "of", "on", "or", "the", "to", "what", "when", "which", "with", "you", "your");

    @Override
    public Response<List<Embedding>> embedAll(List<TextSegment> segments) {
        return Response.from(segments.stream().map(segment -> embedText(segment.text())).toList());
    }

    @Override
    public int dimension() {
        return DIMENSIONS;
    }

    /**
     * Hashes the words of {@code text} into a unit vector. Text with no usable words maps to a fixed vector, so it
     * is equally (and weakly) similar to everything rather than producing an undefined cosine.
     */
    static Embedding embedText(String text) {
        float[] vector = new float[DIMENSIONS];
        for (String word : words(text)) {
            int hash = word.hashCode();
            // The sign comes from a different part of the hash, so two words that share a bucket can cancel out
            // instead of always adding up; that keeps collisions from looking like similarity.
            vector[Math.floorMod(hash, DIMENSIONS)] += ((hash >>> 16) & 1) == 0 ? 1f : -1f;
        }
        double norm = Math.sqrt(sumOfSquares(vector));
        if (norm == 0) {
            Arrays.fill(vector, (float) (1 / Math.sqrt(DIMENSIONS)));
            return Embedding.from(vector);
        }
        for (int i = 0; i < vector.length; i++) {
            vector[i] = (float) (vector[i] / norm);
        }
        return Embedding.from(vector);
    }

    /** Lower-cased words without stop words, with a trailing plural "s" removed so "days" matches "day". */
    static List<String> words(String text) {
        return Arrays.stream(text.toLowerCase(Locale.ROOT).split("[^\\p{L}\\p{N}]+"))
                .filter(word -> word.length() > 1 && !STOP_WORDS.contains(word))
                .map(word -> word.length() > 3 && word.endsWith("s") && !word.endsWith("ss")
                        ? word.substring(0, word.length() - 1) : word)
                .toList();
    }

    private static double sumOfSquares(float[] vector) {
        double sum = 0;
        for (float x : vector) {
            sum += x * x;
        }
        return sum;
    }
}
