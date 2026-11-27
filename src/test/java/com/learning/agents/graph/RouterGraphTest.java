package com.learning.agents.graph;

import com.learning.agents.MainRunner;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class RouterGraphTest {

    private static RouterState route(String query) throws Exception {
        return RouterGraph.build(RouterGraph.placeholder()).invoke(Map.of(RouterState.QUERY, query)).orElseThrow();
    }

    @Test
    void sendsEachQueryToItsSpecialist() throws Exception {
        assertEquals(RouterGraph.BILLING, route("I was charged twice").route());
        assertTrue(route("I was charged twice").answer().orElseThrow().contains("refund"));
        assertEquals(RouterGraph.TECHNICAL, route("error 500").route());
        assertEquals(RouterGraph.GENERAL, route("hello").route());
    }

    @Test
    void specialistSeesItsOwnInstructions() throws Exception {
        RouterState result = RouterGraph.build(prompt -> prompt.startsWith("Classify") ? "technical" : prompt)
                .invoke(Map.of(RouterState.QUERY, "help"))
                .orElseThrow();
        assertTrue(result.answer().orElseThrow().startsWith(RouterGraph.SPECIALISTS.get(RouterGraph.TECHNICAL)));
    }

    @Test
    void unexpectedClassifierReplyFallsBackToGeneral() {
        assertEquals(RouterGraph.GENERAL, RouterGraph.normalizeRoute("I'm not sure, maybe sales?"));
        assertEquals(RouterGraph.BILLING, RouterGraph.normalizeRoute("Category: BILLING."));
    }

    @Test
    void mainRoutesTheDemoQueries() throws Exception {
        String out = MainRunner.run(RouterGraph::main);
        assertTrue(out.contains("billing") && out.contains("technical") && out.contains("general"));
        assertTrue(MainRunner.run(RouterGraph::main, "hello", "there").contains("hello there"));
    }
}
