package com.learning.agents.langgraph;

import com.learning.agents.llm.LlmProvider;
import com.learning.agents.llm.ScriptedLlm;
import com.learning.agents.llm.StreamingLlm;
import org.bsc.langgraph4j.CompiledGraph;
import org.bsc.langgraph4j.GraphInput;
import org.bsc.langgraph4j.GraphStateException;
import org.bsc.langgraph4j.NodeOutput;
import org.bsc.langgraph4j.RunnableConfig;
import org.bsc.langgraph4j.StateGraph;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import java.util.function.Function;

import static org.bsc.langgraph4j.StateGraph.END;
import static org.bsc.langgraph4j.StateGraph.START;
import static org.bsc.langgraph4j.action.AsyncNodeAction.node_async;

/**
 * Streaming at two levels, so the user sees progress instead of a long silence:
 * <ol>
 *   <li><b>Node events</b>: {@code graph.stream(...)} returns each node's output as soon as that node finishes,
 *       instead of {@code invoke} returning only the final state. Good for progress bars and logs ("planning…",
 *       "answering…").</li>
 *   <li><b>Tokens</b>: inside the {@code answer} node, a {@link StreamingLlm} hands each piece of the reply to a
 *       callback as the model produces it. With Ollama this is {@code OllamaStreamingChatModel}; with the placeholder,
 *       the canned reply is split into words.</li>
 * </ol>
 * Note that LangGraph4j also emits bookkeeping events such as {@code __START__} and {@code __END__}; real UIs filter
 * them out, as {@link #isUserNode} does.
 *
 * <pre>
 *   START → plan ──(event: plan)──→ answer ──(tokens…, then event: answer)──→ END
 * </pre>
 */
public class StreamingAgent {

    /**
     * Builds the graph.
     *
     * @param llm       a normal model, for the planning step
     * @param streaming a streaming model, for the answer
     * @param onToken   receives each piece of the answer as it arrives
     */
    public static CompiledGraph<StreamingState> build(Function<String, String> llm, StreamingLlm streaming,
                                                      Consumer<String> onToken) throws GraphStateException {
        return new StateGraph<>(StreamingState.SCHEMA, StreamingState::new)
                .addNode("plan", node_async(state -> Map.of(StreamingState.KEY_POINTS, llm.apply(
                        "List the two key points a good answer must cover, on one line.\nQuestion: " + state.question()).strip())))
                .addNode("answer", node_async(state -> Map.of(StreamingState.ANSWER, streaming.stream("""
                        Answer in three sentences, covering these points: %s
                        Question: %s""".formatted(state.keyPoints(), state.question()), onToken).strip())))
                .addEdge(START, "plan")
                .addEdge("plan", "answer")
                .addEdge("answer", END)
                .compile();
    }

    /**
     * Runs the graph with {@code stream} instead of {@code invoke}, calling {@code onNode} after each node.
     *
     * @return the names of the user-defined nodes, in the order they finished
     */
    static List<String> run(CompiledGraph<StreamingState> graph, String question, Consumer<NodeOutput<StreamingState>> onNode) {
        List<String> finished = new ArrayList<>();
        for (NodeOutput<StreamingState> output : graph.stream(GraphInput.args(Map.of(StreamingState.QUESTION, question)),
                RunnableConfig.builder().build())) {
            if (isUserNode(output.node())) {
                finished.add(output.node());
                onNode.accept(output);
            }
        }
        return finished;
    }

    /** Filters out LangGraph4j's own events ({@code __START__}, {@code __END__}, ...). */
    static boolean isUserNode(String node) {
        return !(node.startsWith("__") && node.endsWith("__"));
    }

    /** Canned replies so the sample runs without a model. */
    static ScriptedLlm placeholder() {
        return ScriptedLlm.create()
                .when("two key points", "what streaming is; why users prefer it")
                .when("Answer in three sentences",
                        "Streaming sends the reply piece by piece as the model writes it. "
                                + "Users see the first words in well under a second instead of waiting for the whole answer. "
                                + "That makes slow local models feel much faster.");
    }

    /** Runs the graph on a question (or {@code args}), printing answer tokens as they stream and one event per finished node. */
    public static void main(String[] args) throws Exception {
        String question = args.length > 0 ? String.join(" ", args) : "Why do chat apps stream their answers?";
        ScriptedLlm canned = placeholder();
        var graph = build(LlmProvider.resolve(canned), LlmProvider.resolveStreaming(StreamingLlm.fromReplies(canned)),
                token -> {
                    System.out.print(token);
                    System.out.flush();
                });

        System.out.println("Question: " + question);
        run(graph, question, output -> System.out.println(switch (output.node()) {
            case "plan" -> "[event] plan finished: " + output.state().keyPoints() + "\n[event] answer streaming:";
            case "answer" -> "\n[event] answer finished (" + output.state().answer().orElse("").length() + " chars)";
            default -> "[event] " + output.node();
        }));
    }
}
