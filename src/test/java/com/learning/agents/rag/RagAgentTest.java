package com.learning.agents.rag;

import com.learning.agents.MainRunner;
import com.learning.agents.rag.KeywordRetriever.Doc;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class RagAgentTest {

    private final KeywordRetriever handbook = new KeywordRetriever(RagAgent.HANDBOOK);

    @Test
    void retrievesTheMostRelevantDocumentFirst() {
        var hits = handbook.search("How many days of paid leave?", 2);

        assertEquals("leave-policy", hits.getFirst().doc().id());
        assertTrue(hits.size() <= 2);
        assertTrue(hits.getFirst().score() > 0 && hits.getFirst().score() <= 1.0 + 1e-9);
    }

    @Test
    void unrelatedQueryRetrievesNothing() {
        assertTrue(handbook.search("office wifi password", 3).isEmpty());
        assertTrue(handbook.search("the and of", 3).isEmpty());
        assertTrue(new KeywordRetriever(List.of()).search("leave", 3).isEmpty());
    }

    @Test
    void rareWordsOutweighCommonOnes() {
        var retriever = new KeywordRetriever(List.of(
                new Doc("a", "policy policy policy apples"),
                new Doc("b", "policy bananas"),
                new Doc("c", "policy cherries")));
        assertEquals("b", retriever.search("policy bananas", 1).getFirst().doc().id());
    }

    @Test
    void promptContainsContextCitationsAndRefusalInstruction() {
        var hits = handbook.search("remote work", 1);
        String prompt = RagAgent.prompt("Can I work remotely?", hits);

        assertTrue(prompt.contains("<doc id=\"remote-work\">"));
        assertTrue(prompt.contains(RagAgent.NO_ANSWER));
        assertTrue(prompt.endsWith("Question: Can I work remotely?"));
    }

    @Test
    void skipsTheModelWhenNothingIsRetrieved() {
        var calls = new AtomicInteger();
        var agent = new RagAgent(handbook, prompt -> {
            calls.incrementAndGet();
            return "made up";
        });

        var answer = agent.ask("What is the capital of Peru?");

        assertEquals(RagAgent.NO_ANSWER, answer.text());
        assertEquals(0, calls.get());
    }

    @Test
    void answerCitesItsSource() {
        var answer = new RagAgent(handbook, RagAgent.placeholder()).ask("How many days of leave do I get?");

        assertTrue(answer.text().contains("[leave-policy]"));
        assertFalse(answer.sources().isEmpty());
    }

    @Test
    void tokenizeDropsStopWordsAndPunctuation() {
        assertEquals(List.of("laptops", "replaced", "3y"), KeywordRetriever.tokenize("Are laptops replaced, a 3y?"));
    }

    @Test
    void mainAnswersAndRefuses() throws Exception {
        String out = MainRunner.run(RagAgent::main);
        assertTrue(out.contains("[leave-policy]") && out.contains(RagAgent.NO_ANSWER));
        assertTrue(MainRunner.run(RagAgent::main, "laptops").contains("Q: laptops"));
    }
}
