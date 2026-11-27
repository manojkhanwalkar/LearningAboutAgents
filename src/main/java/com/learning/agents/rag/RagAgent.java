package com.learning.agents.rag;

import com.learning.agents.llm.LlmProvider;
import com.learning.agents.llm.ScriptedLlm;
import com.learning.agents.rag.KeywordRetriever.Doc;
import com.learning.agents.rag.KeywordRetriever.Hit;

import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * Retrieval-augmented generation (RAG): instead of trusting what the model memorised, look up relevant documents
 * first and put them in the prompt. The model is told to answer only from that context and to cite document ids,
 * so every claim can be checked. If retrieval finds nothing relevant, the agent says so <em>without calling the
 * LLM</em>, the simplest defence against a confident made-up answer.
 *
 * <pre>
 *   question → retrieve top-k docs ─┬─ none relevant → "I don't know" (no LLM call)
 *                                   └─ context + question → LLM → answer with [doc-id] citations
 * </pre>
 */
public class RagAgent {

    static final int TOP_K = 2;
    static final String NO_ANSWER = "I don't know: none of the documents cover that.";

    /** The knowledge base. In a real system this is loaded from files or a database and indexed once. */
    static final List<Doc> HANDBOOK = List.of(
            new Doc("leave-policy", "Employees get 25 days of paid leave per year. Unused leave of up to 5 days carries over to the next year."),
            new Doc("expenses", "Submit expense claims within 30 days with a receipt. Meals while travelling are covered up to 40 euros per day."),
            new Doc("remote-work", "Staff may work remotely up to 3 days per week. Core hours are 10:00 to 15:00."),
            new Doc("laptops", "Laptops are replaced every 3 years. Report lost or stolen laptops to IT within 24 hours."));

    /** The answer plus the documents it was grounded in. */
    record Answer(String text, List<Hit> sources) {
    }

    private final KeywordRetriever retriever;
    private final Function<String, String> llm;

    RagAgent(KeywordRetriever retriever, Function<String, String> llm) {
        this.retriever = retriever;
        this.llm = llm;
    }

    /** Retrieves up to {@link #TOP_K} documents and answers from them; returns {@link #NO_ANSWER} without an LLM call if none match. */
    public Answer ask(String question) {
        List<Hit> hits = retriever.search(question, TOP_K);
        if (hits.isEmpty()) {
            return new Answer(NO_ANSWER, hits);
        }
        return new Answer(llm.apply(prompt(question, hits)).strip(), hits);
    }

    /** Builds the grounded prompt: the retrieved documents in {@code <doc id>} tags, citation and refusal rules, then the question. */
    static String prompt(String question, List<Hit> hits) {
        String context = hits.stream()
                .map(hit -> "<doc id=\"%s\">%s</doc>".formatted(hit.doc().id(), hit.doc().text()))
                .collect(Collectors.joining("\n"));
        return """
                Answer the question using ONLY the documents below. Cite the id of each document you use, like [doc-id].
                If the documents do not contain the answer, reply exactly: %s
                %s
                Question: %s""".formatted(NO_ANSWER, context, question);
    }

    /** Canned replies so the sample runs without a model. */
    static ScriptedLlm placeholder() {
        return ScriptedLlm.create()
                .when("Question: How many days of leave", "You get 25 days of paid leave a year, and up to 5 unused days carry over [leave-policy].")
                .when("Question: Can I work from home", "Yes, up to 3 days per week, as long as you keep core hours of 10:00 to 15:00 [remote-work].")
                .otherwise(NO_ANSWER);
    }

    public static void main(String[] args) {
        var agent = new RagAgent(new KeywordRetriever(HANDBOOK), LlmProvider.resolve(placeholder()));
        var questions = args.length > 0
                ? List.of(String.join(" ", args))
                : List.of("How many days of leave do I get?", "Can I work from home on Fridays?", "What is the office wifi password?");

        for (String question : questions) {
            Answer answer = agent.ask(question);
            System.out.println("Q: " + question);
            System.out.println("Retrieved: " + answer.sources().stream()
                    .map(hit -> "%s (%.2f)".formatted(hit.doc().id(), hit.score()))
                    .collect(Collectors.joining(", ", "[", "]")));
            System.out.println("A: " + answer.text());
            System.out.println();
        }
    }
}
