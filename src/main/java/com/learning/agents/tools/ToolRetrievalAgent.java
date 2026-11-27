package com.learning.agents.tools;

import com.learning.agents.llm.HashingEmbeddingModel;
import com.learning.agents.llm.LlmProvider;
import com.learning.agents.llm.Replies;
import com.learning.agents.llm.ScriptedLlm;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.CosineSimilarity;

import java.time.LocalDate;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.function.Function;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Tool retrieval: when an agent has dozens of tools, listing all of them in every prompt wastes tokens and confuses
 * small models, which then pick the wrong one. So treat the tool catalog like a RAG corpus: embed each tool's
 * description once, and for each question show the model only the {@link #TOP_K} most relevant tools. The model then
 * picks one with {@code CALL tool(input)}, and code runs it, but only if it was one of the tools shown.
 *
 * <pre>
 *   question → embed → nearest tool descriptions (top 3 of 12) → model picks one → run it → answer
 * </pre>
 */
public class ToolRetrievalAgent {

    static final int TOP_K = 3;

    /** Applied to the line after {@code CALL}: a tool name and its input up to the <em>last</em> closing parenthesis. */
    private static final Pattern CALL_ARGS = Pattern.compile("^(\\w+)\\((.*)\\)\\s*$");

    /** A tool: its name, the description the model and the retriever both read, and what it does. */
    record Tool(String name, String description, Function<String, String> run) {
    }

    /** What happened: which tools were shown, which was called with what, and the result. */
    record Outcome(List<String> shown, Optional<String> call, String result) {
    }

    /** A catalog bigger than you'd want in one prompt. The descriptions are what retrieval matches against. */
    static final List<Tool> CATALOG = List.of(
            new Tool("word_count", "Count the words in a piece of text.", s -> String.valueOf(s.isBlank() ? 0 : s.strip().split("\\s+").length)),
            new Tool("reverse_text", "Reverse the characters of a piece of text.", s -> new StringBuilder(s).reverse().toString()),
            new Tool("uppercase", "Convert text to upper case letters.", s -> s.toUpperCase(Locale.ROOT)),
            new Tool("km_to_miles", "Convert a distance in kilometres to miles.", s -> String.format(Locale.ROOT, "%.2f", Double.parseDouble(s.strip()) * 0.621371)),
            new Tool("kg_to_pounds", "Convert a weight in kilograms to pounds.", s -> String.format(Locale.ROOT, "%.2f", Double.parseDouble(s.strip()) * 2.20462)),
            new Tool("celsius_to_fahrenheit", "Convert a temperature in Celsius to Fahrenheit.", s -> String.format(Locale.ROOT, "%.1f", Double.parseDouble(s.strip()) * 9 / 5 + 32)),
            new Tool("day_of_week", "Give the day of the week for a date in YYYY-MM-DD format.", s -> LocalDate.parse(s.strip()).getDayOfWeek().toString()),
            new Tool("is_leap_year", "Check whether a year is a leap year.", s -> String.valueOf(java.time.Year.isLeap(Long.parseLong(s.strip())))),
            new Tool("vowel_count", "Count the vowels in a piece of text.", s -> String.valueOf(s.toLowerCase(Locale.ROOT).chars().filter(c -> "aeiou".indexOf(c) >= 0).count())),
            new Tool("palindrome_check", "Check whether a word reads the same backwards (a palindrome).", s -> String.valueOf(new StringBuilder(s.strip()).reverse().toString().equalsIgnoreCase(s.strip()))),
            new Tool("square_root", "Compute the square root of a number.", s -> String.format(Locale.ROOT, "%.4f", Math.sqrt(Double.parseDouble(s.strip())))),
            new Tool("percent_of", "Compute a percentage of a number, input like '15% of 80'.", s -> {
                var m = Pattern.compile("([\\d.]+)%\\s*of\\s*([\\d.]+)").matcher(s);
                return m.find() ? String.format(Locale.ROOT, "%.2f", Double.parseDouble(m.group(1)) / 100 * Double.parseDouble(m.group(2))) : "error: use 'N% of M'";
            }));

    private record Indexed(Tool tool, Embedding embedding) {
    }

    private final List<Indexed> index;
    private final EmbeddingModel embeddings;
    private final Function<String, String> llm;

    /** Embeds every tool description once, up front. */
    ToolRetrievalAgent(List<Tool> catalog, EmbeddingModel embeddings, Function<String, String> llm) {
        this.embeddings = embeddings;
        this.llm = llm;
        this.index = catalog.stream()
                .map(tool -> new Indexed(tool, embeddings.embed(TextSegment.from(tool.name() + ": " + tool.description())).content()))
                .toList();
    }

    /** Returns the {@code k} tools whose descriptions are most similar to the question, best first. */
    List<Tool> retrieve(String question, int k) {
        if (question.isBlank()) {
            return List.of();
        }
        Embedding q = embeddings.embed(question).content();
        return index.stream()
                .sorted(Comparator.comparingDouble((Indexed i) -> CosineSimilarity.between(q, i.embedding())).reversed())
                .limit(k)
                .map(Indexed::tool)
                .toList();
    }

    /** Shows the model only the retrieved tools, runs the one it picks, and returns what happened. */
    public Outcome ask(String question) {
        List<Tool> shown = retrieve(question, TOP_K);
        String reply = llm.apply("""
                Pick the one tool that answers the question and reply with exactly: CALL <tool>(<input>)
                If none fits, reply: NONE
                Tools:
                %s
                Question: %s""".formatted(shown.stream().map(t -> "- %s: %s".formatted(t.name(), t.description()))
                .collect(Collectors.joining("\n")), question));
        List<String> names = shown.stream().map(Tool::name).toList();
        var call = Replies.lineAfter(reply, "CALL ").map(CALL_ARGS::matcher).orElse(null);
        if (call == null || !call.find()) {
            return new Outcome(names, Optional.empty(), "No suitable tool among " + names + ".");
        }
        String name = call.group(1);
        String input = call.group(2).replaceAll("^['\"]|['\"]$", "");
        Optional<Tool> tool = shown.stream().filter(t -> t.name().equals(name)).findFirst();
        if (tool.isEmpty()) {
            // The model may only use what it was shown; a name it remembers or invents is refused.
            return new Outcome(names, Optional.of(name + "(" + input + ")"), "error: tool '" + name + "' was not offered");
        }
        String result;
        try {
            result = tool.get().run().apply(input);
        } catch (RuntimeException e) {
            result = "error: bad input for " + name + " (" + e.getClass().getSimpleName() + ")";
        }
        return new Outcome(names, Optional.of(name + "(" + input + ")"), result);
    }

    /** Canned replies: picks the matching tool from the ones shown. */
    static ScriptedLlm placeholder() {
        return ScriptedLlm.create()
                .when(p -> p.contains("- km_to_miles") && p.contains("Question: How far is 42"), p -> "CALL km_to_miles(42.195)")
                .when(p -> p.contains("- day_of_week") && p.contains("2026-12-25"), p -> "CALL day_of_week(2026-12-25)")
                .when(p -> p.contains("- palindrome_check") && p.contains("racecar"), p -> "CALL palindrome_check(racecar)")
                .otherwise("NONE");
    }

    public static void main(String[] args) {
        var agent = new ToolRetrievalAgent(CATALOG, LlmProvider.resolveEmbeddings(new HashingEmbeddingModel()),
                LlmProvider.resolve(placeholder()));
        var questions = args.length > 0
                ? List.of(String.join(" ", args))
                : List.of("How far is 42.195 kilometres in miles?",
                "Which day of the week is the date 2026-12-25?",
                "Is the word racecar a palindrome?");

        System.out.println("Catalog: " + CATALOG.size() + " tools; showing the model the top " + TOP_K + " per question.");
        for (String question : questions) {
            Outcome outcome = agent.ask(question);
            System.out.println("Q: " + question);
            System.out.println("  shown:  " + outcome.shown());
            System.out.println("  called: " + outcome.call().orElse("(nothing)") + " → " + outcome.result());
        }
    }
}
