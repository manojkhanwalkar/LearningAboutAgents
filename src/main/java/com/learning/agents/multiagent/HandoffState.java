package com.learning.agents.multiagent;

import org.bsc.langgraph4j.state.AgentState;
import org.bsc.langgraph4j.state.Channel;
import org.bsc.langgraph4j.state.Channels;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * State shared by the agents of {@link HandoffGraph}.
 */
public class HandoffState extends AgentState {

    public static final String REQUEST = "request";
    public static final String ACTIVE = "active";
    public static final String HANDOFFS = "handoffs";
    public static final String TRAIL = "trail";
    public static final String ANSWER = "answer";

    /** {@code trail} records every turn in order (duplicates kept, an agent can be visited twice). */
    public static final Map<String, Channel<?>> SCHEMA = Map.of(
            TRAIL, Channels.appenderWithDuplicate(ArrayList::new)
    );

    public HandoffState(Map<String, Object> initData) {
        super(initData);
    }

    public String request() {
        return this.<String>value(REQUEST).orElseThrow();
    }

    /** The agent that should act next; the entry agent until someone hands off. */
    public String active() {
        return this.<String>value(ACTIVE).orElse(HandoffGraph.ENTRY);
    }

    public int handoffs() {
        return this.<Integer>value(HANDOFFS).orElse(0);
    }

    /** One line per turn, such as {@code "triage → billing (refund question)"}. */
    public List<String> trail() {
        return this.<List<String>>value(TRAIL).orElse(List.of());
    }

    public Optional<String> answer() {
        return value(ANSWER);
    }
}
