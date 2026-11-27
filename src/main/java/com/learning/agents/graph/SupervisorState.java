package com.learning.agents.graph;

import org.bsc.langgraph4j.state.AgentState;
import org.bsc.langgraph4j.state.Channel;
import org.bsc.langgraph4j.state.Channels;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * State shared by the supervisor and worker agents of {@link SupervisorGraph}.
 */
public class SupervisorState extends AgentState {

    public static final String REQUEST = "request";
    public static final String NEXT = "next";
    public static final String TURNS = "turns";
    public static final String WORK = "work";

    /**
     * {@code work} accumulates "agent: output" entries, the shared scratchpad; other keys are overwritten. Duplicates
     * are kept, because an agent repeating itself is part of the history the supervisor should see.
     */
    public static final Map<String, Channel<?>> SCHEMA = Map.of(
            WORK, Channels.appenderWithDuplicate(ArrayList::new)
    );

    public SupervisorState(Map<String, Object> initData) {
        super(initData);
    }

    public String request() {
        return this.<String>value(REQUEST).orElseThrow();
    }

    /** The agent the supervisor picked, or {@link SupervisorGraph#FINISH}. */
    public String next() {
        return this.<String>value(NEXT).orElse(SupervisorGraph.FINISH);
    }

    /** How many times the supervisor has made a decision. */
    public int turns() {
        return this.<Integer>value(TURNS).orElse(0);
    }

    /** Every agent contribution so far, each formatted as {@code "agent: output"}. */
    public List<String> work() {
        return this.<List<String>>value(WORK).orElse(List.of());
    }
}
