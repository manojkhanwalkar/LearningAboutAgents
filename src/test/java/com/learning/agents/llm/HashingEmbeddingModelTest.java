package com.learning.agents.llm;

import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.CosineSimilarity;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class HashingEmbeddingModelTest {

    private final HashingEmbeddingModel model = new HashingEmbeddingModel();

    private double similarity(String a, String b) {
        return CosineSimilarity.between(model.embed(a).content(), model.embed(b).content());
    }

    @Test
    void sharedWordsAreMoreSimilarThanUnrelatedText() {
        double related = similarity("unused leave carries over", "How much leave carries over?");
        double unrelated = similarity("unused leave carries over", "football results tonight");
        assertTrue(related > 0.5, "related was " + related);
        assertTrue(unrelated < related);
    }

    @Test
    void vectorsAreUnitLengthWithTheDeclaredDimension() {
        var embedding = model.embed("hello world").content();
        assertEquals(HashingEmbeddingModel.DIMENSIONS, embedding.dimension());
        assertEquals(HashingEmbeddingModel.DIMENSIONS, model.dimension());
        double norm = 0;
        for (float x : embedding.vector()) {
            norm += x * x;
        }
        assertEquals(1.0, norm, 1e-5);
    }

    @Test
    void textWithNoWordsStillGetsAUsableVector() {
        // LangChain4j rejects blank segments before they reach a model, so call embedText directly for "".
        var empty = HashingEmbeddingModel.embedText("");
        assertEquals(1.0, CosineSimilarity.between(empty, model.embed("the of and").content()), 1e-5);
        assertTrue(Double.isFinite(CosineSimilarity.between(empty, model.embed("leave").content())));
    }

    @Test
    void embedAllKeepsOrderAndWordsDropsStopWordsAndPlurals() {
        var all = model.embedAll(List.of(TextSegment.from("days"), TextSegment.from("day"))).content();
        assertEquals(1.0, CosineSimilarity.between(all.get(0), all.get(1)), 1e-5);
        assertEquals(List.of("day", "glass", "bus"), HashingEmbeddingModel.words("The days, the glass, a bus"));
    }
}
