package com.learning.agents.chat;

import com.learning.agents.llm.LlmProvider;
import com.learning.agents.llm.ScriptedLlm;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * Conversation memory with summarisation. An LLM is stateless: it only "remembers" what you resend in the prompt.
 * Resending everything eventually overflows the context window, so this agent keeps the last
 * {@link #RECENT_TURNS} turns word for word and folds older turns into a running summary written by the LLM.
 * Each prompt is then: system instructions + summary + recent turns + the new message.
 * <p>
 * LangChain4j ships ready-made memories ({@code MessageWindowChatMemory}, {@code TokenWindowChatMemory}). This
 * sample builds one by hand so you can see what goes into the prompt on every turn.
 */
public class ConversationMemoryAgent {

    static final int RECENT_TURNS = 4;

    /** One message in the conversation. */
    record Turn(String role, String text) {
        @Override
        public String toString() {
            return role + ": " + text;
        }
    }

    private final Function<String, String> llm;
    private final Deque<Turn> recent = new ArrayDeque<>();
    private String summary = "";

    ConversationMemoryAgent(Function<String, String> llm) {
        this.llm = llm;
    }

    /** Sends one user message with the remembered context, stores both sides, and returns the reply. */
    public String chat(String userMessage) {
        String reply = llm.apply(prompt(userMessage));
        remember(new Turn("User", userMessage));
        remember(new Turn("Assistant", reply));
        return reply;
    }

    String prompt(String userMessage) {
        return """
                You are a helpful assistant. Use the conversation so far to answer.
                Summary of earlier conversation: %s
                Recent messages:
                %s
                User: %s
                Assistant:""".formatted(summary.isBlank() ? "(none)" : summary, render(recent), userMessage);
    }

    /** Adds a turn; once more than {@link #RECENT_TURNS} are held, the oldest two are summarised away. */
    private void remember(Turn turn) {
        recent.addLast(turn);
        if (recent.size() > RECENT_TURNS) {
            var evicted = List.of(recent.removeFirst(), recent.removeFirst());
            summary = llm.apply("""
                    Update the running summary of a conversation with the new messages.
                    Keep every fact the user stated (names, preferences, numbers). Reply with the summary only, max 60 words.
                    Current summary: %s
                    New messages:
                    %s""".formatted(summary.isBlank() ? "(none)" : summary, render(evicted))).strip();
        }
    }

    String summary() {
        return summary;
    }

    List<Turn> recent() {
        return List.copyOf(recent);
    }

    private static String render(Iterable<Turn> turns) {
        var text = new StringBuilder();
        turns.forEach(turn -> text.append(turn).append('\n'));
        return text.toString().strip();
    }

    /** Canned replies so the sample runs without a model. The last answer only works if the summary kept the name. */
    static ScriptedLlm placeholder() {
        return ScriptedLlm.create()
                .when("Update the running summary", "The user is Priya, a Java developer who prefers short answers.")
                .when(prompt -> prompt.endsWith("What's my name?\nAssistant:") && prompt.contains("Priya"),
                        prompt -> "Your name is Priya.")
                .when("What's my name?", "Sorry, I don't remember.")
                .otherwise("Got it.");
    }

    public static void main(String[] args) {
        var agent = new ConversationMemoryAgent(LlmProvider.resolve(placeholder()));
        var messages = args.length > 0
                ? List.of(String.join(" ", args))
                : List.of("Hi, I'm Priya.", "I write Java for a living.", "Please keep answers short.",
                "Which build tool should I learn?", "What's my name?");

        for (String message : messages) {
            System.out.println("User     : " + message);
            System.out.println("Assistant: " + agent.chat(message));
            System.out.println("  memory → summary: " + (agent.summary().isBlank() ? "(none)" : agent.summary())
                    + " | recent turns: " + agent.recent().size());
        }
        System.out.println("\nRecent turns kept verbatim:\n" + agent.recent().stream()
                .map(Turn::toString).collect(Collectors.joining("\n")));
    }
}
