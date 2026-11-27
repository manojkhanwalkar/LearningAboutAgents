package com.learning.agents.llm;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.function.Predicate;

/**
 * A placeholder LLM that answers from a script instead of calling a model. Each rule pairs a test on the
 * prompt with a reply, and the first rule that matches wins. If no rule matches, the fallback reply is used.
 * <p>
 * The samples use it so they run with no Ollama server, and the tests use it as a fake. Instances are
 * immutable: {@link #when} and {@link #otherwise} return a new copy, so one instance can be shared across threads.
 *
 * <pre>{@code
 * var llm = ScriptedLlm.create()
 *         .when("Classify", "billing")
 *         .otherwise("I don't know.");
 * }</pre>
 */
public final class ScriptedLlm implements Function<String, String> {

    /** One scripted behaviour: if {@code matches} accepts the prompt, {@code reply} produces the answer. */
    private record Rule(Predicate<String> matches, Function<String, String> reply) {
    }

    private final List<Rule> rules;
    private final String fallback;

    private ScriptedLlm(List<Rule> rules, String fallback) {
        this.rules = List.copyOf(rules);
        this.fallback = fallback;
    }

    /** Starts an empty script whose fallback reply is {@code "(placeholder reply)"}. */
    public static ScriptedLlm create() {
        return new ScriptedLlm(List.of(), "(placeholder reply)");
    }

    /** Replies with {@code reply} when the prompt contains {@code fragment}. */
    public ScriptedLlm when(String fragment, String reply) {
        return when(prompt -> prompt.contains(fragment), prompt -> reply);
    }

    /** Replies with {@code reply.apply(prompt)} when {@code matches} accepts the prompt. */
    public ScriptedLlm when(Predicate<String> matches, Function<String, String> reply) {
        var next = new ArrayList<>(rules);
        next.add(new Rule(matches, reply));
        return new ScriptedLlm(next, fallback);
    }

    /** Sets the reply used when no rule matches. */
    public ScriptedLlm otherwise(String reply) {
        return new ScriptedLlm(rules, reply);
    }

    @Override
    public String apply(String prompt) {
        return rules.stream()
                .filter(rule -> rule.matches().test(prompt))
                .findFirst()
                .map(rule -> rule.reply().apply(prompt))
                .orElse(fallback);
    }
}
