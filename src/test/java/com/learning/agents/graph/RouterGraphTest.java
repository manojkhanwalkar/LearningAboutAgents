package com.learning.agents.graph;

import com.learning.agents.MainRunner;
import org.bsc.langgraph4j.GraphInput;
import org.bsc.langgraph4j.RunnableConfig;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class RouterGraphTest {

    private static RouterState route(String query) throws Exception {
        return RouterGraph.build(RouterGraph.placeholder()).invoke(GraphInput.args(Map.of(RouterState.QUERY, query)), RunnableConfig.builder().build()).orElseThrow();
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
                .invoke(GraphInput.args(Map.of(RouterState.QUERY, "help")), RunnableConfig.builder().build())
                .orElseThrow();
        assertTrue(result.answer().orElseThrow().startsWith(RouterGraph.SPECIALISTS.get(RouterGraph.TECHNICAL)));
    }

    @Test
    void unexpectedClassifierReplyFallsBackToGeneral() {
        assertEquals(RouterGraph.GENERAL, RouterGraph.normalizeRoute("I'm not sure, maybe sales?"));
        assertEquals(RouterGraph.BILLING, RouterGraph.normalizeRoute("Category: BILLING."));
        assertEquals(RouterGraph.GENERAL, RouterGraph.normalizeRoute("technical (not billing)"));
    }

    @Test
    void mainRoutesTheDemoQueries() throws Exception {
        String out = MainRunner.run(RouterGraph::main);
        assertTrue(out.contains("billing") && out.contains("technical") && out.contains("general"));
        assertTrue(MainRunner.run(RouterGraph::main, "hello", "there").contains("hello there"));
    }
}
