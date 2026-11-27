package com.learning.agents.langgraph;

import com.learning.agents.MainRunner;
import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import org.bsc.langgraph4j.GraphInput;
import org.bsc.langgraph4j.RunnableConfig;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for the LangGraph4j feature samples: parallel branches, subgraphs, time travel and streaming.
 */
class LangGraphFeaturesTest {

    // ── Parallel branches ──────────────────────────────────────────────────────────────────────────────────────

    @Test
    void everyBranchContributesAndTheJudgeSeesAllViews() throws Exception {
        List<String> judgePrompts = Collections.synchronizedList(new ArrayList<>());
        var graph = ParallelBranchesGraph.build(prompt -> {
            if (prompt.startsWith("Weigh")) {
                judgePrompts.add(prompt);
            }
            return ParallelBranchesGraph.placeholder().apply(prompt);
        });

        ParallelState result = ParallelBranchesGraph.run(graph, "q", null);

        assertEquals(3, result.views().size());
        assertEquals(1, judgePrompts.size());
        assertTrue(judgePrompts.getFirst().contains("optimist:") && judgePrompts.getFirst().contains("skeptic:"));
        assertTrue(result.verdict().orElseThrow().startsWith("Run locally"));
    }

    @Test
    void withAnExecutorTheBranchesReallyOverlap() throws Exception {
        // Each branch waits until all three are running at once; that can only happen if they run in parallel.
        var allStarted = new CountDownLatch(ParallelBranchesGraph.ANALYSTS.size());
        var graph = ParallelBranchesGraph.build(prompt -> {
            if (!prompt.startsWith("Weigh")) {
                allStarted.countDown();
                try {
                    if (!allStarted.await(5, TimeUnit.SECONDS)) {
                        return "ran alone";
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
            return "together";
        });

        ParallelState result;
        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            result = ParallelBranchesGraph.run(graph, "q", executor);
        }

        assertEquals(3, result.views().size(), "no branch update may be lost when branches run in parallel");
        assertTrue(result.views().stream().allMatch(view -> view.endsWith("together")), result.views().toString());
    }

    @Test
    void withoutAnExecutorTheBranchesRunOneAtATime() throws Exception {
        var running = new AtomicInteger();
        var maxRunning = new AtomicInteger();
        var graph = ParallelBranchesGraph.build(prompt -> {
            maxRunning.accumulateAndGet(running.incrementAndGet(), Math::max);
            try {
                Thread.sleep(50);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            running.decrementAndGet();
            return "x";
        });

        ParallelBranchesGraph.run(graph, "q", null);

        assertEquals(1, maxRunning.get());
    }

    @Test
    void latencyWrapperDelaysAndPropagatesInterrupts() {
        long start = System.nanoTime();
        assertEquals("r", ParallelBranchesGraph.withLatency(p -> "r", Duration.ofMillis(30)).apply("p"));
        assertTrue(System.nanoTime() - start >= 30_000_000);

        Thread.currentThread().interrupt();
        try {
            assertThrows(IllegalStateException.class,
                    () -> ParallelBranchesGraph.withLatency(p -> "r", Duration.ofSeconds(5)).apply("p"));
            assertTrue(Thread.interrupted());
        } finally {
            Thread.interrupted();
        }
    }

    @Test
    void parallelMainPrintsBothTimings() throws Exception {
        String out = MainRunner.run(ParallelBranchesGraph::main);
        assertTrue(out.contains("Default (branches one after another)") && out.contains("With a parallel executor"));
    }

    // ── Subgraphs ──────────────────────────────────────────────────────────────────────────────────────────────

    @Test
    void parentRunsTheSubgraphAsOneStepAndUsesItsOutput() throws Exception {
        ReportState result = SubgraphGraph.run(SubgraphGraph.build(SubgraphGraph.placeholder()), "topic");

        assertEquals(List.of("outline: done", "write: done"), result.log());
        assertEquals(List.of("gather round 1: +2 note(s)", "gather round 2: +1 note(s)", "summarize: done"), result.researchLog());
        assertEquals(3, result.notes().size());
        assertTrue(result.report().orElseThrow().startsWith("Small open models"));
    }

    @Test
    void subgraphWorksOnItsOwnAndStopsAtMaxRounds() throws Exception {
        var gathers = new AtomicInteger();
        ReportState result = SubgraphGraph.research(prompt -> {
                    if (prompt.startsWith("You are a researcher")) {
                        gathers.incrementAndGet();
                        return "";   // never finds anything
                    }
                    return "nothing found";
                }).compile()
                .invoke(GraphInput.args(Map.of(ReportState.TOPIC, "t", ReportState.OUTLINE, "o")), RunnableConfig.builder().build())
                .orElseThrow();

        assertEquals(SubgraphGraph.MAX_ROUNDS, gathers.get());
        assertEquals("nothing found", result.summary());
        assertTrue(result.notes().isEmpty());
    }

    @Test
    void subgraphMainPrintsBothLogs() throws Exception {
        String out = MainRunner.run(SubgraphGraph::main);
        assertTrue(out.contains("[report] outline: done") && out.contains("[research] summarize: done"));
        assertEquals(1, out.split("\\[report] outline: done", -1).length - 1, "parent log lines must not be duplicated");
    }

    // ── Time travel ────────────────────────────────────────────────────────────────────────────────────────────

    @Test
    void historyListsEveryCheckpointOldestFirst() throws Exception {
        var agent = new TimeTravelGraph(TimeTravelGraph.placeholder(), "t1");
        agent.run("an app");

        var history = agent.history();
        assertEquals(List.of("__START__", "draft", "polish"), history.stream().map(s -> s.node()).toList());
        assertEquals("draft", history.getFirst().nextNodeId());
    }

    @Test
    void forkingFromACheckpointChangesTheOutcomeAndKeepsTheOriginal() throws Exception {
        var agent = new TimeTravelGraph(TimeTravelGraph.placeholder(), "t2");
        String original = agent.run("an app").tagline().orElseThrow();
        String beforeDraft = agent.checkpointBefore("draft").checkpointId();

        String replayed = agent.replay(beforeDraft).tagline().orElseThrow();
        String forked = agent.fork(beforeDraft, Map.of(TaglineState.TONE, "playful")).tagline().orElseThrow();

        assertEquals(original, replayed);
        assertTrue(forked.contains("brain"));
        assertFalse(original.contains("brain"));
        // The step we forked from must survive the fork: it is still in history, unchanged, and can be replayed again.
        var survivor = agent.history().stream().filter(s -> s.checkpointId().equals(beforeDraft)).findFirst().orElseThrow();
        assertEquals("friendly", survivor.state().tone());
        assertEquals(original, agent.replay(beforeDraft).tagline().orElseThrow());
        assertTrue(agent.fork(beforeDraft, Map.of(TaglineState.TONE, "playful")).tagline().orElseThrow().contains("brain"));
    }

    @Test
    void replayReRunsTheLaterNodes() throws Exception {
        var calls = new AtomicInteger();
        var agent = new TimeTravelGraph(prompt -> "v" + calls.incrementAndGet(), "t3");
        agent.run("x");                                          // draft = v1, polish = v2
        String beforePolish = agent.checkpointBefore("polish").checkpointId();

        assertEquals("v3", agent.replay(beforePolish).tagline().orElseThrow());
        assertThrows(IllegalArgumentException.class, () -> agent.checkpointBefore("nope"));
    }

    @Test
    void timeTravelMainShowsHistoryReplayAndFork() throws Exception {
        String out = MainRunner.run(TimeTravelGraph::main);
        assertTrue(out.contains("History (oldest first)") && out.contains("Fork with tone = \"playful\": Notes so handy"));
    }

    // ── Streaming ──────────────────────────────────────────────────────────────────────────────────────────────

    @Test
    void nodeEventsArriveInOrderAndTokensAddUpToTheAnswer() throws Exception {
        List<String> tokens = new ArrayList<>();
        List<String> events = new ArrayList<>();
        var canned = StreamingAgent.placeholder();
        var graph = StreamingAgent.build(canned, com.learning.agents.llm.StreamingLlm.fromReplies(canned), tokens::add);

        List<String> finished = StreamingAgent.run(graph, "q", output -> events.add(output.node()));

        assertEquals(List.of("plan", "answer"), finished);
        assertEquals(finished, events);
        assertTrue(tokens.size() > 10);
        assertTrue(String.join("", tokens).startsWith("Streaming sends the reply"));
    }

    @Test
    void frameworkEventsAreFilteredOut() {
        assertFalse(StreamingAgent.isUserNode("__START__"));
        assertFalse(StreamingAgent.isUserNode("__END__"));
        assertTrue(StreamingAgent.isUserNode("plan"));
        assertTrue(StreamingAgent.isUserNode("__half"));
    }

    @Test
    void streamingMainPrintsEventsAndTokens() throws Exception {
        String out = MainRunner.run(StreamingAgent::main);
        assertTrue(out.contains("[event] plan finished") && out.contains("[event] answer finished"));
    }
}
