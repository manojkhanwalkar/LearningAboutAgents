package com.learning.agents.graph;

import org.bsc.langgraph4j.state.AgentState;
import org.bsc.langgraph4j.state.Channel;
import org.bsc.langgraph4j.state.Channels;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * State shared by the nodes of {@link OrchestratorWorkersGraph}.
 */
public class OrchestratorState extends AgentState {

    public static final String TASK = "task";
    public static final String SUBTASKS = "subtasks";
    public static final String NEXT_INDEX = "nextIndex";
    public static final String RESULTS = "results";
    public static final String REPORT = "report";

    /**
     * {@code results} accumulates one worker output per subtask; other keys are overwritten. It must keep duplicates
     * ({@code Channels.appender} silently drops a value equal to one already in the list), or two workers that
     * happen to reply with the same text would leave {@code results} shorter than {@code subtasks}.
     */
    public static final Map<String, Channel<?>> SCHEMA = Map.of(
            RESULTS, Channels.appenderWithDuplicate(ArrayList::new)
    );

    public OrchestratorState(Map<String, Object> initData) {
        super(initData);
    }

    public String task() {
        return this.<String>value(TASK).orElseThrow();
    }

    /** The plan the orchestrator wrote, one subtask per entry. */
    public List<String> subtasks() {
        return this.<List<String>>value(SUBTASKS).orElse(List.of());
    }

    /** Index into {@link #subtasks()} of the next subtask a worker should take. */
    public int nextIndex() {
        return this.<Integer>value(NEXT_INDEX).orElse(0);
    }

    /** Worker outputs, in the same order as {@link #subtasks()}. */
    public List<String> results() {
        return this.<List<String>>value(RESULTS).orElse(List.of());
    }

    public Optional<String> report() {
        return value(REPORT);
    }
}
