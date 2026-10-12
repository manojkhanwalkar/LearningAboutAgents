package com.learning.agents.graph;

import org.bsc.langgraph4j.state.AgentState;
import org.bsc.langgraph4j.state.Channel;
import org.bsc.langgraph4j.state.Channels;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * State shared by the nodes of {@link ReActAgentGraph}.
 */
public class ReActState extends AgentState {

    public static final String QUESTION = "question";
    public static final String TRANSCRIPT = "transcript";
    public static final String ACTION = "action";
    public static final String ACTION_INPUT = "actionInput";
    public static final String ANSWER = "answer";
    public static final String STEPS = "steps";

    /** {@code transcript} accumulates Thought/Action/Observation entries; other keys are overwritten. */
    public static final Map<String, Channel<?>> SCHEMA = Map.of(
            TRANSCRIPT, Channels.appender(ArrayList::new)
    );

    public ReActState(Map<String, Object> initData) {
        super(initData);
    }

    public String question() {
        return this.<String>value(QUESTION).orElseThrow();
    }

    /** Every Thought/Action and Observation so far, in order. Fed back to the model on each step. */
    public List<String> transcript() {
        return this.<List<String>>value(TRANSCRIPT).orElse(List.of());
    }

    /** The tool chosen by the latest reasoning step; empty string if the model's reply could not be parsed. */
    public String action() {
        return this.<String>value(ACTION).orElse("");
    }

    public String actionInput() {
        return this.<String>value(ACTION_INPUT).orElse("");
    }

    public Optional<String> answer() {
        return value(ANSWER);
    }

    public int steps() {
        return this.<Integer>value(STEPS).orElse(0);
    }
}
