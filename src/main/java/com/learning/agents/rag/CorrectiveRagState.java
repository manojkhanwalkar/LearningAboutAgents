package com.learning.agents.rag;

import com.learning.agents.rag.VectorRetriever.Chunk;
import org.bsc.langgraph4j.state.AgentState;
import org.bsc.langgraph4j.state.Channel;
import org.bsc.langgraph4j.state.Channels;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * State shared by the nodes of {@link CorrectiveRagGraph}.
 */
public class CorrectiveRagState extends AgentState {

    public static final String QUESTION = "question";
    public static final String QUERY = "query";
    public static final String RETRIEVED = "retrieved";
    public static final String RELEVANT = "relevant";
    public static final String REWRITES = "rewrites";
    public static final String ANSWER = "answer";
    public static final String LOG = "log";

    /** {@code log} accumulates one line per step (duplicates kept, since a retry can repeat a step); others are overwritten. */
    public static final Map<String, Channel<?>> SCHEMA = Map.of(
            LOG, Channels.appenderWithDuplicate(ArrayList::new)
    );

    public CorrectiveRagState(Map<String, Object> initData) {
        super(initData);
    }

    public String question() {
        return this.<String>value(QUESTION).orElseThrow();
    }

    /** The search query in use: the question itself at first, then each rewrite. */
    public String query() {
        return this.<String>value(QUERY).orElseGet(this::question);
    }

    /** What the latest search returned, before grading. */
    public List<Chunk> retrieved() {
        return this.<List<Chunk>>value(RETRIEVED).orElse(List.of());
    }

    /** The retrieved chunks the grader judged relevant. */
    public List<Chunk> relevant() {
        return this.<List<Chunk>>value(RELEVANT).orElse(List.of());
    }

    public int rewrites() {
        return this.<Integer>value(REWRITES).orElse(0);
    }

    public Optional<String> answer() {
        return value(ANSWER);
    }

    public List<String> log() {
        return this.<List<String>>value(LOG).orElse(List.of());
    }
}
