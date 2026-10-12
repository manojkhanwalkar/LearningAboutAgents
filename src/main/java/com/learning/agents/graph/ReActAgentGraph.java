package com.learning.agents.graph;

import com.learning.agents.OllamaConfig;
import com.learning.agents.graph.ReActTools.ReActTool;
import dev.langchain4j.model.chat.ChatModel;
import org.bsc.langgraph4j.CompiledGraph;
import org.bsc.langgraph4j.GraphStateException;
import org.bsc.langgraph4j.StateGraph;

import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import static org.bsc.langgraph4j.StateGraph.END;
import static org.bsc.langgraph4j.StateGraph.START;
import static org.bsc.langgraph4j.action.AsyncEdgeAction.edge_async;
import static org.bsc.langgraph4j.action.AsyncNodeAction.node_async;

/**
 * The ReAct pattern (Reason + Act, Yao et al. 2022) with every step out in the open: the model writes a
 * {@code Thought} and an {@code Action: tool[input]} as plain text, the graph runs the tool and appends an
 * {@code Observation}, and the loop repeats until the model writes a {@code Final Answer} or
 * {@link #MAX_STEPS} is reached. Unlike {@code tools.ToolCallingAgent}, which uses native tool calls, nothing
 * here is hidden inside the framework, so you can see exactly what the model is shown on each step.
 *
 * <pre>
 *   START → reason ─┬─ Final Answer / step limit → END
 *              ↑    │
 *              │    └─ Action → act
 *              └──── Observation ──┘
 * </pre>
 */
public class ReActAgentGraph {

    static final int MAX_STEPS = 6;

    private static final Pattern THOUGHT = Pattern.compile("Thought:\\s*(.*?)\\s*(?=Action:|Final Answer:|$)", Pattern.DOTALL);
    private static final Pattern ACTION = Pattern.compile("Action:\\s*(\\w+)\\s*\\[(.*?)]", Pattern.DOTALL);
    private static final Pattern FINAL_ANSWER = Pattern.compile("Final Answer:\\s*(.*)", Pattern.DOTALL);

    /** One parsed model reply. Exactly one of {@code action} or {@code answer} is set when the reply is well-formed. */
    record Step(String thought, String action, String input, String answer) {
        boolean isFinal() {
            return answer != null;
        }
    }

    public static CompiledGraph<ReActState> build(Function<String, String> llm) throws GraphStateException {
        return build(llm, ReActTools.defaults());
    }

    /**
     * Builds the graph around a pluggable LLM and tool set so it can be tested without Ollama.
     *
     * @param llm   maps a prompt to a model reply
     * @param tools the actions the model may take
     */
    public static CompiledGraph<ReActState> build(Function<String, String> llm, List<ReActTool> tools)
            throws GraphStateException {
        Map<String, ReActTool> byName = tools.stream()
                .collect(Collectors.toMap(ReActTool::name, Function.identity()));

        return new StateGraph<>(ReActState.SCHEMA, ReActState::new)
                // Reason: show the model the question plus the transcript so far, and parse its next step.
                .addNode("reason", node_async(state -> {
                    String reply = truncateAtObservation(llm.apply(prompt(state, tools)));
                    Step step = parse(reply);
                    int steps = state.steps() + 1;
                    if (step.isFinal()) {
                        return Map.of(
                                ReActState.TRANSCRIPT, reply,
                                ReActState.ANSWER, step.answer(),
                                ReActState.STEPS, steps);
                    }
                    return Map.of(
                            ReActState.TRANSCRIPT, reply,
                            ReActState.ACTION, step.action() == null ? "" : step.action(),
                            ReActState.ACTION_INPUT, step.input() == null ? "" : step.input(),
                            ReActState.STEPS, steps);
                }))
                // Act: run the chosen tool and record what it returned as an Observation.
                .addNode("act", node_async(state -> {
                    ReActTool tool = byName.get(state.action());
                    String observation = tool == null
                            ? "Invalid action. Reply with 'Action: <tool>[<input>]' using one of: %s, or give a 'Final Answer:'."
                                    .formatted(String.join(", ", byName.keySet()))
                            : tool.fn().apply(state.actionInput());
                    return Map.of(ReActState.TRANSCRIPT, "Observation: " + observation);
                }))
                .addEdge(START, "reason")
                .addConditionalEdges("reason",
                        edge_async(ReActAgentGraph::route),
                        Map.of("act", "act", "finish", END))
                .addEdge("act", "reason")
                .compile();
    }

    static String route(ReActState state) {
        return state.answer().isPresent() || state.steps() >= MAX_STEPS ? "finish" : "act";
    }

    static String prompt(ReActState state, List<ReActTool> tools) {
        String toolList = tools.stream()
                .map(t -> "- %s: %s".formatted(t.name(), t.description()))
                .collect(Collectors.joining("\n"));
        return """
                Answer the question by reasoning step by step and using tools.
                Available tools:
                %s

                On each turn reply with exactly ONE step in this format:
                Thought: <what you know so far and what to do next>
                Action: <tool>[<input>]

                When you have enough information, reply instead with:
                Thought: <your final reasoning>
                Final Answer: <the answer>

                Never write an Observation yourself; it will be given to you after each Action.

                Question: %s
                %s"""
                .formatted(toolList, state.question(), String.join("\n", state.transcript()));
    }

    /** Models often keep going and invent their own Observation; everything from there on is discarded. */
    static String truncateAtObservation(String reply) {
        int i = reply.indexOf("Observation:");
        return (i >= 0 ? reply.substring(0, i) : reply).strip();
    }

    static Step parse(String reply) {
        var thought = THOUGHT.matcher(reply);
        String t = thought.find() ? thought.group(1).strip() : "";
        var answer = FINAL_ANSWER.matcher(reply);
        if (answer.find()) {
            return new Step(t, null, null, answer.group(1).strip());
        }
        var action = ACTION.matcher(reply);
        if (action.find()) {
            return new Step(t, action.group(1).strip(), action.group(2).strip(), null);
        }
        return new Step(t, null, null, null);
    }

    public static void main(String[] args) throws Exception {
        ChatModel model = OllamaConfig.chatModel();
        String question = args.length > 0
                ? String.join(" ", args)
                : "How many times taller is the Burj Khalifa than the Eiffel Tower?";

        ReActState result = build(model::chat)
                .invoke(Map.of(ReActState.QUESTION, question))
                .orElseThrow();

        System.out.println("Question: " + question);
        System.out.println();
        result.transcript().forEach(entry -> System.out.println(entry + "\n"));
        System.out.println(result.answer()
                .map(a -> "Answer after " + result.steps() + " reasoning step(s): " + a)
                .orElse("No final answer within " + MAX_STEPS + " steps."));
    }
}
