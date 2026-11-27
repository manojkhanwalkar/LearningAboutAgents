package com.learning.agents.graph;

import org.bsc.langgraph4j.state.AgentState;
import org.bsc.langgraph4j.state.Channel;

import java.util.Map;
import java.util.Optional;

/**
 * State shared by the nodes of {@link HumanApprovalGraph}.
 */
public class ApprovalState extends AgentState {

    public static final String REQUEST = "request";
    public static final String PROPOSAL = "proposal";
    public static final String APPROVED = "approved";
    public static final String RESULT = "result";

    /** Every key is overwritten, so no channels are needed. */
    public static final Map<String, Channel<?>> SCHEMA = Map.of();

    public ApprovalState(Map<String, Object> initData) {
        super(initData);
    }

    public String request() {
        return this.<String>value(REQUEST).orElseThrow();
    }

    /** The action the agent wants to take, waiting for a human decision. */
    public String proposal() {
        return this.<String>value(PROPOSAL).orElse("");
    }

    /** The human's decision; {@code false} until someone approves, so the safe default is "don't act". */
    public boolean approved() {
        return this.<Boolean>value(APPROVED).orElse(false);
    }

    /** Present only once the graph has finished, whether the action ran or was cancelled. */
    public Optional<String> result() {
        return value(RESULT);
    }
}
