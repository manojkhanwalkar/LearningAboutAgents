package com.learning.agents.rag;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.io.Serializable;
import java.util.List;
import java.util.Map;

/**
 * The embedding half of RAG, using LangChain4j's own building blocks: documents are split into overlapping chunks
 * ({@link DocumentSplitters#recursive}), each chunk is embedded, and the vectors go into an
 * {@link InMemoryEmbeddingStore}. A search embeds the query the same way and returns the nearest chunks.
 * Swapping in a persistent store (pgvector, Qdrant, Chroma...) only changes the store line.
 * Not thread-safe while indexing; searches after {@link #index} are safe from several threads.
 */
public final class VectorRetriever {

    /**
     * One retrieved chunk: its text, the id of the document it came from, and its relevance score. Serializable
     * because LangGraph4j copies graph state with Java serialization, so every value kept in state must be.
     */
    public record Chunk(String text, String source, double score) implements Serializable {
    }

    static final int CHUNK_CHARS = 300;
    static final int OVERLAP_CHARS = 40;
    static final String SOURCE = "source";

    private final EmbeddingModel embeddings;
    private final InMemoryEmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
    private final DocumentSplitter splitter = DocumentSplitters.recursive(CHUNK_CHARS, OVERLAP_CHARS);

    public VectorRetriever(EmbeddingModel embeddings) {
        this.embeddings = embeddings;
    }

    /** Splits, embeds and stores each document; the map key becomes the citation id of its chunks. */
    public VectorRetriever index(Map<String, String> documentsById) {
        documentsById.forEach((id, text) -> {
            List<TextSegment> chunks = splitter.split(Document.from(text, Metadata.from(SOURCE, id)));
            store.addAll(embeddings.embedAll(chunks).content(), chunks);
        });
        return this;
    }

    /**
     * Returns up to {@code maxResults} chunks scoring at least {@code minScore}, best first. Scores are LangChain4j
     * relevance scores, {@code (cosine + 1) / 2}: 1 is identical, 0.5 is unrelated, so useful thresholds sit above 0.5.
     */
    public List<Chunk> search(String query, int maxResults, double minScore) {
        var request = EmbeddingSearchRequest.builder()
                .queryEmbedding(embeddings.embed(query).content())
                .maxResults(maxResults)
                .minScore(minScore)
                .build();
        return store.search(request).matches().stream()
                .map(match -> new Chunk(match.embedded().text(), match.embedded().metadata().getString(SOURCE), match.score()))
                .toList();
    }
}
