package com.learning.agents.langgraph;

import org.bsc.langgraph4j.state.AgentState;
import org.bsc.langgraph4j.state.Channel;

import java.util.Map;
import java.util.Optional;

/**
 * State shared by the nodes of {@link StreamingAgent}.
 */
public class StreamingState extends AgentState {

    public static final String QUESTION = "question";
    public static final String KEY_POINTS = "keyPoints";
    public static final String ANSWER = "answer";

    /** Every key is overwritten, so no channels are needed. */
    public static final Map<String, Channel<?>> SCHEMA = Map.of();

    public StreamingState(Map<String, Object> initData) {
        super(initData);
    }

    public String question() {
        return this.<String>value(QUESTION).orElseThrow();
    }

    public String keyPoints() {
        return this.<String>value(KEY_POINTS).orElse("");
    }

    public Optional<String> answer() {
        return value(ANSWER);
    }
}
