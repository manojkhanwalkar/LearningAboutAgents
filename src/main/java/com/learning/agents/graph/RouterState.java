package com.learning.agents.graph;

import org.bsc.langgraph4j.state.AgentState;
import org.bsc.langgraph4j.state.Channel;

import java.util.Map;
import java.util.Optional;

/**
 * State shared by the nodes of {@link RouterGraph}.
 */
public class RouterState extends AgentState {

    public static final String QUERY = "query";
    public static final String ROUTE = "route";
    public static final String ANSWER = "answer";

    /** Every key is overwritten, so no channels are needed. */
    public static final Map<String, Channel<?>> SCHEMA = Map.of();

    public RouterState(Map<String, Object> initData) {
        super(initData);
    }

    public String query() {
        return this.<String>value(QUERY).orElseThrow();
    }

    /** The specialist the classifier picked; {@code general} until the classifier has run. */
    public String route() {
        return this.<String>value(ROUTE).orElse(RouterGraph.GENERAL);
    }

    public Optional<String> answer() {
        return value(ANSWER);
    }
}
