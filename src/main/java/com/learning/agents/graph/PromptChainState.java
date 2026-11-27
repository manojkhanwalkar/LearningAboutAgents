package com.learning.agents.graph;

import org.bsc.langgraph4j.state.AgentState;
import org.bsc.langgraph4j.state.Channel;
import org.bsc.langgraph4j.state.Channels;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * State shared by the nodes of {@link PromptChainingGraph}.
 */
public class PromptChainState extends AgentState {

    public static final String TOPIC = "topic";
    public static final String OUTLINE = "outline";
    public static final String OUTLINE_ATTEMPTS = "outlineAttempts";
    public static final String DRAFT = "draft";
    public static final String FINAL_TEXT = "finalText";
    public static final String LOG = "log";

    /** {@code log} accumulates one line per step; other keys are overwritten. */
    public static final Map<String, Channel<?>> SCHEMA = Map.of(
            LOG, Channels.appender(ArrayList::new)
    );

    public PromptChainState(Map<String, Object> initData) {
        super(initData);
    }

    public String topic() {
        return this.<String>value(TOPIC).orElseThrow();
    }

    public String outline() {
        return this.<String>value(OUTLINE).orElse("");
    }

    public int outlineAttempts() {
        return this.<Integer>value(OUTLINE_ATTEMPTS).orElse(0);
    }

    public Optional<String> draft() {
        return value(DRAFT);
    }

    /** The polished text, present only when every step of the chain ran. */
    public Optional<String> finalText() {
        return value(FINAL_TEXT);
    }

    public List<String> log() {
        return this.<List<String>>value(LOG).orElse(List.of());
    }
}
