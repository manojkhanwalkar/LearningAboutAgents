package com.learning.agents.rag;

import com.learning.agents.llm.HashingEmbeddingModel;
import com.learning.agents.rag.VectorRetriever.Chunk;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class VectorRetrieverTest {

    /** Counts embedAll calls and the segments in each, delegating to the placeholder embeddings. */
    private static final class CountingEmbeddings implements EmbeddingModel {
        final List<Integer> batchSizes = new ArrayList<>();
        private final HashingEmbeddingModel delegate = new HashingEmbeddingModel();

        @Override
        public Response<List<Embedding>> embedAll(List<TextSegment> segments) {
            batchSizes.add(segments.size());
            return delegate.embedAll(segments);
        }
    }

    /** The original indexing: one embedAll call per document, added in document order. */
    private static List<Chunk> referenceSearch(Map<String, String> docs, String query, int max, double minScore) {
        var embeddings = new HashingEmbeddingModel();
        var store = new InMemoryEmbeddingStore<TextSegment>();
        var splitter = dev.langchain4j.data.document.splitter.DocumentSplitters.recursive(
                VectorRetriever.CHUNK_CHARS, VectorRetriever.OVERLAP_CHARS);
        docs.forEach((id, text) -> {
            List<TextSegment> chunks = splitter.split(Document.from(text, Metadata.from(VectorRetriever.SOURCE, id)));
            store.addAll(embeddings.embedAll(chunks).content(), chunks);
        });
        var request = EmbeddingSearchRequest.builder().queryEmbedding(embeddings.embed(query).content())
                .maxResults(max).minScore(minScore).build();
        return store.search(request).matches().stream()
                .map(m -> new Chunk(m.embedded().text(), m.embedded().metadata().getString(VectorRetriever.SOURCE), m.score()))
                .toList();
    }

    @Test
    void searchResultsMatchPerDocumentIndexing() {
        var retriever = new VectorRetriever(new HashingEmbeddingModel()).index(EmbeddingRagAgent.HANDBOOK);
        for (String query : List.of("unused paid leave days carry over next year", "USB drives unknown sources",
                "password manager", "expenses receipts", "football")) {
            assertEquals(referenceSearch(EmbeddingRagAgent.HANDBOOK, query, 10, 0.0), retriever.search(query, 10, 0.0), query);
        }
    }

    @Test
    void indexEmbedsChunksOfManyDocumentsInFewBatches() {
        Map<String, String> docs = new LinkedHashMap<>();
        for (int i = 0; i < 100; i++) {
            docs.put("doc" + i, "Document number " + i + " talks about topic " + i + ".");
        }
        var embeddings = new CountingEmbeddings();
        var retriever = new VectorRetriever(embeddings).index(docs);

        // 100 one-chunk documents: 4 batched calls (32 + 32 + 32 + 4), not 100 calls of one segment each.
        assertEquals(List.of(32, 32, 32, 4), embeddings.batchSizes);
        assertEquals(List.of("doc42"), retriever.search("topic 42", 1, 0.0).stream().map(Chunk::source).toList());
    }

    @Test
    void emptyCorpusMakesNoEmbeddingCall() {
        var embeddings = new CountingEmbeddings();
        var retriever = new VectorRetriever(embeddings).index(Map.of());

        assertTrue(embeddings.batchSizes.isEmpty());
        assertTrue(retriever.search("anything", 3, 0.0).isEmpty());
    }
}
