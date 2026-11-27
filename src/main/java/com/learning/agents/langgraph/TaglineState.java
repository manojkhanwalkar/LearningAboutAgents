package com.learning.agents.langgraph;

import org.bsc.langgraph4j.state.AgentState;
import org.bsc.langgraph4j.state.Channel;

import java.util.Map;
import java.util.Optional;

/**
 * State shared by the nodes of {@link TimeTravelGraph}.
 */
public class TaglineState extends AgentState {

    public static final String PRODUCT = "product";
    public static final String TONE = "tone";
    public static final String DRAFT = "draft";
    public static final String TAGLINE = "tagline";

    /** Every key is overwritten, so no channels are needed. */
    public static final Map<String, Channel<?>> SCHEMA = Map.of();

    public TaglineState(Map<String, Object> initData) {
        super(initData);
    }

    public String product() {
        return this.<String>value(PRODUCT).orElseThrow();
    }

    /** The style the draft is written in; {@code friendly} unless set. Changing it in a fork changes the outcome. */
    public String tone() {
        return this.<String>value(TONE).orElse("friendly");
    }

    public String draft() {
        return this.<String>value(DRAFT).orElse("");
    }

    public Optional<String> tagline() {
        return value(TAGLINE);
    }
}
