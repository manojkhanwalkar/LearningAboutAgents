package com.learning.agents.rag;

import com.learning.agents.MainRunner;
import com.learning.agents.llm.HashingEmbeddingModel;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EmbeddingRagAgentTest {

    private final VectorRetriever handbook = new VectorRetriever(new HashingEmbeddingModel()).index(EmbeddingRagAgent.HANDBOOK);

    @Test
    void vectorSearchFindsTheRightChunkAndHonoursTheThreshold() {
        var hits = handbook.search("unused leave carry over", 2, EmbeddingRagAgent.MIN_VECTOR_SCORE);
        assertEquals("leave-policy", hits.getFirst().source());
        assertTrue(hits.getFirst().score() > EmbeddingRagAgent.MIN_VECTOR_SCORE);
        assertTrue(handbook.search("football results", 4, EmbeddingRagAgent.MIN_VECTOR_SCORE).isEmpty());
    }

    @Test
    void longDocumentsAreSplitIntoSeveralChunks() {
        var chunks = new VectorRetriever(new HashingEmbeddingModel())
                .index(Map.of("security", EmbeddingRagAgent.HANDBOOK.get("security")))
                .search("security", 10, 0.0);
        assertTrue(chunks.size() > 1, "expected the long page to be chunked");
        assertTrue(chunks.stream().allMatch(c -> c.text().length() <= VectorRetriever.CHUNK_CHARS));
    }

    @Test
    void rewritesThenRerankedChunksGroundTheAnswer() {
        List<String> prompts = new ArrayList<>();
        var agent = new EmbeddingRagAgent(handbook, prompt -> {
            prompts.add(prompt);
            return EmbeddingRagAgent.placeholder().apply(prompt);
        });

        var answer = agent.ask("Hey, quick one: if I don't use all my leave this year, do I lose it?");

        assertEquals("unused paid leave days carry over next year", answer.query());
        assertEquals("leave-policy", answer.sources().getFirst().chunk().source());
        assertTrue(answer.text().contains("[leave-policy]"));
        assertTrue(prompts.getLast().contains("<doc source=\"leave-policy\">"));
    }

    @Test
    void lowRerankScoresMeanNoAnswerCall() {
        List<String> prompts = new ArrayList<>();
        var agent = new EmbeddingRagAgent(handbook, prompt -> {
            prompts.add(prompt);
            return prompt.startsWith("Rewrite") ? "leave carry over" : "3";
        });

        var answer = agent.ask("leave?");

        assertEquals(EmbeddingRagAgent.NO_ANSWER, answer.text());
        assertTrue(answer.sources().isEmpty());
        assertFalse(prompts.stream().anyMatch(p -> p.startsWith("Answer the question")));
    }

    @Test
    void parseScoreReadsTheFirstNumberAndCapsIt() {
        assertEquals(Optional.of(8), EmbeddingRagAgent.parseScore("8/10"));
        assertEquals(Optional.of(10), EmbeddingRagAgent.parseScore("Score: 100"));
        assertEquals(Optional.empty(), EmbeddingRagAgent.parseScore("very relevant"));
        assertEquals(0, new EmbeddingRagAgent(handbook, p -> "unsure")
                .rerank("q", new VectorRetriever.Chunk("t", "s", 0.9)));
    }

    @Test
    void mainAnswersTwoAndRefusesOne() throws Exception {
        String out = MainRunner.run(EmbeddingRagAgent::main);
        assertTrue(out.contains("[leave-policy]") && out.contains("[security]") && out.contains(EmbeddingRagAgent.NO_ANSWER));
        assertTrue(MainRunner.run(EmbeddingRagAgent::main, "laptops").contains("Q: laptops"));
    }
}
