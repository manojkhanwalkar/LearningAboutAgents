package com.learning.agents.chat;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.json.JsonMapper;
import com.learning.agents.llm.LlmProvider;
import com.learning.agents.llm.ScriptedLlm;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.function.Function;

/**
 * Structured output with a validate-and-repair loop. Code that consumes LLM output needs data, not prose, so the
 * model is asked for JSON matching a schema. Models often wrap the JSON in Markdown fences, miss a field or pick an
 * out-of-range value, so the reply is parsed and validated in code. On failure the exact error is sent back and the
 * model gets another try, up to {@link #MAX_ATTEMPTS}.
 * <p>
 * Ollama can also constrain output with a JSON schema ({@code responseFormat} in LangChain4j), and LangChain4j AI
 * Services can return records directly. The loop here is what those features do for you, and it works with any model.
 */
public class StructuredOutputAgent {

    static final int MAX_ATTEMPTS = 3;
    static final Set<String> SENTIMENTS = Set.of("positive", "neutral", "negative");

    /**
     * Strict on types: by default Jackson would quietly turn {@code 4.9} or {@code "5"} into an int, so a wrong-typed
     * reply would pass validation instead of being sent back for repair. Unknown extra fields are ignored.
     */
    private static final ObjectMapper JSON = JsonMapper.builder()
            .disable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES)
            .enable(DeserializationFeature.FAIL_ON_NULL_FOR_PRIMITIVES)
            .disable(DeserializationFeature.ACCEPT_FLOAT_AS_INT)
            .disable(MapperFeature.ALLOW_COERCION_OF_SCALARS)
            .build();

    /** The structure we want out of a free-text product review. */
    public record ReviewSummary(String product, int rating, String sentiment) {
    }

    /** The parsed value, if any, and every reply the model gave on the way (useful for seeing the repairs). */
    record Extraction(Optional<ReviewSummary> value, List<String> attempts) {
    }

    /**
     * Asks for the review summary as JSON and retries with the validation error until it parses or
     * {@link #MAX_ATTEMPTS} calls have been made.
     *
     * @return the summary if any attempt was valid, plus every raw reply in order
     */
    static Extraction extract(Function<String, String> llm, String review) {
        String task = """
                Extract a summary of the product review below as JSON with exactly these fields:
                {"product": string, "rating": integer 1-5, "sentiment": "positive" | "neutral" | "negative"}
                Reply with the JSON object only.
                Review: <<<%s>>>""".formatted(review);

        String prompt = task;
        List<String> attempts = new ArrayList<>();
        for (int i = 0; i < MAX_ATTEMPTS; i++) {
            String reply = llm.apply(prompt);
            attempts.add(reply);
            try {
                return new Extraction(Optional.of(parse(reply)), attempts);
            } catch (IllegalArgumentException e) {
                // The model is stateless, so the repair prompt repeats the whole task (schema and review) and adds the
                // exact error, so the model knows what to fix and still has the source text to fix it from.
                prompt = """
                        %s

                        Your previous reply was invalid: %s
                        Previous reply: %s
                        Reply again with only the corrected JSON object.""".formatted(task, e.getMessage(), reply);
            }
        }
        return new Extraction(Optional.empty(), attempts);
    }

    /**
     * Parses and validates one reply.
     *
     * @throws IllegalArgumentException with a message suitable to send back to the model
     */
    static ReviewSummary parse(String reply) {
        ReviewSummary summary;
        try {
            summary = JSON.readValue(stripFences(reply), ReviewSummary.class);
        } catch (JsonProcessingException e) {
            throw new IllegalArgumentException("not valid JSON for the schema (" + e.getOriginalMessage() + ")");
        }
        if (summary.product() == null || summary.product().isBlank()) {
            throw new IllegalArgumentException("\"product\" is missing or empty");
        }
        if (summary.rating() < 1 || summary.rating() > 5) {
            throw new IllegalArgumentException("\"rating\" must be between 1 and 5, got " + summary.rating());
        }
        if (summary.sentiment() == null || !SENTIMENTS.contains(summary.sentiment())) {
            throw new IllegalArgumentException("\"sentiment\" must be one of " + SENTIMENTS + ", got " + summary.sentiment());
        }
        return summary;
    }

    /** Removes the ```json ... ``` wrapper many models add, and anything outside the outermost braces. */
    static String stripFences(String reply) {
        int start = reply.indexOf('{');
        int end = reply.lastIndexOf('}');
        return start >= 0 && end > start ? reply.substring(start, end + 1) : reply.strip();
    }

    /** Canned replies so the sample runs without a model: the first answer is wrong, the repair fixes it. */
    static ScriptedLlm placeholder() {
        return ScriptedLlm.create()
                .when("previous reply was invalid", """
                        {"product": "TrailRunner 2 shoes", "rating": 4, "sentiment": "positive"}""")
                .otherwise("""
                        ```json
                        {"product": "TrailRunner 2 shoes", "rating": 8, "sentiment": "positive"}
                        ```""");
    }

    /** Extracts a summary from a sample review (or the text in {@code args}) and prints each attempt and the parsed result. */
    public static void main(String[] args) {
        String review = args.length > 0
                ? String.join(" ", args)
                : "Bought the TrailRunner 2 shoes last month. Super comfy on long runs, though the laces keep coming undone.";

        Extraction result = extract(LlmProvider.resolve(placeholder()), review);

        for (int i = 0; i < result.attempts().size(); i++) {
            System.out.printf("Attempt %d: %s%n", i + 1, result.attempts().get(i).strip());
        }
        System.out.println(result.value()
                .map(summary -> "Parsed: " + summary)
                .orElse("Gave up after " + MAX_ATTEMPTS + " attempts."));
    }
}
