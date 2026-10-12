package com.learning.agents.graph;

import org.bsc.langgraph4j.state.AgentState;
import org.bsc.langgraph4j.state.Channel;
import org.bsc.langgraph4j.state.Channels;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * State shared by the nodes of {@link WriterReviewerGraph}.
 */
public class WriterState extends AgentState {

    public static final String TOPIC = "topic";
    public static final String DRAFT = "draft";
    public static final String FEEDBACK = "feedback";
    public static final String REVISIONS = "revisions";
    public static final String LOG = "log";

    /** {@code log} accumulates entries from every node; other keys are overwritten. */
    public static final Map<String, Channel<?>> SCHEMA = Map.of(
            LOG, Channels.appender(ArrayList::new)
    );

    public WriterState(Map<String, Object> initData) {
        super(initData);
    }

    public String topic() {
        return this.<String>value(TOPIC).orElseThrow();
    }

    public Optional<String> draft() {
        return value(DRAFT);
    }

    public Optional<String> feedback() {
        return value(FEEDBACK);
    }

    public int revisions() {
        return this.<Integer>value(REVISIONS).orElse(0);
    }

    public List<String> log() {
        return this.<List<String>>value(LOG).orElse(List.of());
    }
}
