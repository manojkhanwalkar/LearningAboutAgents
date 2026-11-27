package com.learning.agents.reasoning;

import org.bsc.langgraph4j.state.AgentState;
import org.bsc.langgraph4j.state.Channel;
import org.bsc.langgraph4j.state.Channels;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * State shared by the nodes of {@link ReflexionGraph}.
 */
public class ReflexionState extends AgentState {

    public static final String TASK = "task";
    public static final String ANSWER = "answer";
    public static final String ERROR = "error";
    public static final String ATTEMPTS = "attempts";
    public static final String LESSONS = "lessons";
    public static final String HISTORY = "history";

    /** {@code lessons} and {@code history} accumulate (duplicates kept); the rest are overwritten each attempt. */
    public static final Map<String, Channel<?>> SCHEMA = Map.of(
            LESSONS, Channels.appenderWithDuplicate(ArrayList::new),
            HISTORY, Channels.appenderWithDuplicate(ArrayList::new)
    );

    public ReflexionState(Map<String, Object> initData) {
        super(initData);
    }

    public String task() {
        return this.<String>value(TASK).orElseThrow();
    }

    /** The latest attempt. */
    public String answer() {
        return this.<String>value(ANSWER).orElse("");
    }

    /** Why the latest attempt failed the checker; empty when it passed. */
    public Optional<String> error() {
        return this.<String>value(ERROR).filter(e -> !e.isEmpty());
    }

    public int attempts() {
        return this.<Integer>value(ATTEMPTS).orElse(0);
    }

    /** The lessons written after each failure, oldest first. Passed back into every later attempt. */
    public List<String> lessons() {
        return this.<List<String>>value(LESSONS).orElse(List.of());
    }

    /** One line per attempt: the answer and the checker's verdict. */
    public List<String> history() {
        return this.<List<String>>value(HISTORY).orElse(List.of());
    }

    public boolean passed() {
        return attempts() > 0 && error().isEmpty();
    }
}
