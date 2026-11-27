package com.learning.agents.langgraph;

import org.bsc.langgraph4j.state.AgentState;
import org.bsc.langgraph4j.state.Channel;
import org.bsc.langgraph4j.state.Channels;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * State shared by the parent graph and the research subgraph of {@link SubgraphGraph}. A compiled subgraph used as a
 * node must have the same state type as its parent, so both read and write these keys.
 */
public class ReportState extends AgentState {

    public static final String TOPIC = "topic";
    public static final String OUTLINE = "outline";
    public static final String NOTES = "notes";
    public static final String SUMMARY = "summary";
    public static final String REPORT = "report";
    public static final String LOG = "log";
    public static final String RESEARCH_LOG = "researchLog";
    public static final String ROUNDS = "rounds";

    /**
     * {@code notes}, {@code log} and {@code researchLog} accumulate; the rest are overwritten.
     * <p>
     * When the subgraph finishes, LangGraph4j merges its <em>whole</em> final state back into the parent, so every
     * list it saw on the way in, such as the parent's log lines written before it started, is appended a second time.
     * The plain {@code Channels.appender} drops values that are already in the list, which undoes exactly that, so
     * keys that cross the subgraph boundary use it and their entries are made unique (round numbers, node names).
     * {@code notes} is only written inside the subgraph, which runs once, so it keeps duplicates.
     */
    public static final Map<String, Channel<?>> SCHEMA = Map.of(
            NOTES, Channels.appenderWithDuplicate(ArrayList::new),
            LOG, Channels.appender(ArrayList::new),
            RESEARCH_LOG, Channels.appender(ArrayList::new)
    );

    public ReportState(Map<String, Object> initData) {
        super(initData);
    }

    public String topic() {
        return this.<String>value(TOPIC).orElseThrow();
    }

    public String outline() {
        return this.<String>value(OUTLINE).orElse("");
    }

    /** Raw findings gathered by the research subgraph. */
    public List<String> notes() {
        return this.<List<String>>value(NOTES).orElse(List.of());
    }

    /** The research subgraph's output: its notes condensed. */
    public String summary() {
        return this.<String>value(SUMMARY).orElse("");
    }

    public Optional<String> report() {
        return value(REPORT);
    }

    /** One line per parent node that ran. */
    public List<String> log() {
        return this.<List<String>>value(LOG).orElse(List.of());
    }

    /** One line per subgraph node that ran; written only by the research subgraph. */
    public List<String> researchLog() {
        return this.<List<String>>value(RESEARCH_LOG).orElse(List.of());
    }

    /** How many gather rounds the research subgraph has run. */
    public int rounds() {
        return this.<Integer>value(ROUNDS).orElse(0);
    }
}
