package com.learning.agents.langgraph;

import org.bsc.langgraph4j.state.AgentState;
import org.bsc.langgraph4j.state.Channel;
import org.bsc.langgraph4j.state.Channels;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * State shared by the parallel branches of {@link ParallelBranchesGraph}.
 */
public class ParallelState extends AgentState {

    public static final String QUESTION = "question";
    public static final String VIEWS = "views";
    public static final String VERDICT = "verdict";

    /**
     * {@code views} collects one entry per branch. Branches that run at the same time each return an update, and
     * LangGraph4j merges them through this appender channel, so no branch overwrites another's result.
     */
    public static final Map<String, Channel<?>> SCHEMA = Map.of(
            VIEWS, Channels.appenderWithDuplicate(ArrayList::new)
    );

    public ParallelState(Map<String, Object> initData) {
        super(initData);
    }

    public String question() {
        return this.<String>value(QUESTION).orElseThrow();
    }

    /** One {@code "branch: view"} entry per analyst, in no guaranteed order. */
    public List<String> views() {
        return this.<List<String>>value(VIEWS).orElse(List.of());
    }

    public Optional<String> verdict() {
        return value(VERDICT);
    }
}
