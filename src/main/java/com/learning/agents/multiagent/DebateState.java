package com.learning.agents.multiagent;

import org.bsc.langgraph4j.state.AgentState;
import org.bsc.langgraph4j.state.Channel;
import org.bsc.langgraph4j.state.Channels;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * State shared by the debaters and the judge of {@link DebateGraph}.
 */
public class DebateState extends AgentState {

    public static final String MOTION = "motion";
    public static final String ROUND = "round";
    public static final String TRANSCRIPT = "transcript";
    public static final String WINNER = "winner";
    public static final String REASON = "reason";

    /** {@code transcript} holds every argument in order (duplicates kept, as a debater may repeat a point). */
    public static final Map<String, Channel<?>> SCHEMA = Map.of(
            TRANSCRIPT, Channels.appenderWithDuplicate(ArrayList::new)
    );

    public DebateState(Map<String, Object> initData) {
        super(initData);
    }

    public String motion() {
        return this.<String>value(MOTION).orElseThrow();
    }

    /** Completed rounds; a round is one argument from each side. */
    public int round() {
        return this.<Integer>value(ROUND).orElse(0);
    }

    /** Every argument so far, each as {@code "PRO (round n): ..."} or {@code "CON (round n): ..."}. */
    public List<String> transcript() {
        return this.<List<String>>value(TRANSCRIPT).orElse(List.of());
    }

    /** {@code PRO}, {@code CON}, or {@code UNDECIDED} if the judge's verdict couldn't be read. */
    public Optional<String> winner() {
        return value(WINNER);
    }

    public String reason() {
        return this.<String>value(REASON).orElse("");
    }
}
