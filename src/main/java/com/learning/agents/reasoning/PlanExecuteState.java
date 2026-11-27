package com.learning.agents.reasoning;

import org.bsc.langgraph4j.state.AgentState;
import org.bsc.langgraph4j.state.Channel;
import org.bsc.langgraph4j.state.Channels;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * State shared by the nodes of {@link PlanExecuteGraph}.
 */
public class PlanExecuteState extends AgentState {

    public static final String GOAL = "goal";
    public static final String PLAN = "plan";
    public static final String PAST_STEPS = "pastSteps";
    public static final String PLAN_HISTORY = "planHistory";
    public static final String STEPS = "steps";
    public static final String ANSWER = "answer";

    /** {@code pastSteps} and {@code planHistory} accumulate (duplicates kept); {@code plan} is replaced by each replan. */
    public static final Map<String, Channel<?>> SCHEMA = Map.of(
            PAST_STEPS, Channels.appenderWithDuplicate(ArrayList::new),
            PLAN_HISTORY, Channels.appenderWithDuplicate(ArrayList::new)
    );

    public PlanExecuteState(Map<String, Object> initData) {
        super(initData);
    }

    public String goal() {
        return this.<String>value(GOAL).orElseThrow();
    }

    /** The steps still to do, first one next. */
    public List<String> plan() {
        return this.<List<String>>value(PLAN).orElse(List.of());
    }

    /** Every executed step with its result, as {@code "step → result"}. */
    public List<String> pastSteps() {
        return this.<List<String>>value(PAST_STEPS).orElse(List.of());
    }

    /** Each version of the plan, joined with " | ", so you can see how replanning changed it. */
    public List<String> planHistory() {
        return this.<List<String>>value(PLAN_HISTORY).orElse(List.of());
    }

    public int steps() {
        return this.<Integer>value(STEPS).orElse(0);
    }

    public Optional<String> answer() {
        return value(ANSWER);
    }
}
