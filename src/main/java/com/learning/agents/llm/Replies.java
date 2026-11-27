package com.learning.agents.llm;

import java.util.Optional;

/**
 * Reads labelled lines such as {@code ANSWER: ...} or {@code HANDOFF: billing} out of model replies. Models decorate
 * these lines ({@code **ANSWER:**}, {@code > FINAL:}) and often write a sentence first, so the label may be on any
 * line and may follow Markdown or punctuation.
 * <p>
 * This scans line by line instead of using a multi-line regex like {@code (?m)^\W*ANSWER:}: such a pattern can cross
 * newlines and retries from every line start, which takes quadratic time on a long, mostly blank reply (a model stuck
 * printing newlines). Here every character is looked at a bounded number of times.
 */
public final class Replies {

    private Replies() {
    }

    /**
     * Finds the first line that starts with {@code label} (case-insensitive, after leading decoration) and returns
     * the rest of that line plus every following line, stripped of decoration and whitespace.
     */
    public static Optional<String> after(String reply, String label) {
        return find(reply, label, true);
    }

    /** Like {@link #after}, but returns only the rest of the matching line. */
    public static Optional<String> lineAfter(String reply, String label) {
        return find(reply, label, false);
    }

    /** Shared scan for {@link #after} and {@link #lineAfter}: the text after the first line starting with {@code label}, to the end of the reply or of that line, or empty when there is no such line or nothing follows the label. */
    private static Optional<String> find(String reply, String label, boolean toEnd) {
        int start = 0;
        while (start <= reply.length()) {
            int end = reply.indexOf('\n', start);
            if (end < 0) {
                end = reply.length();
            }
            int i = skipDecoration(reply, start, end);
            if (reply.regionMatches(true, i, label, 0, label.length())) {
                int restStart = skipEmphasis(reply, i + label.length(), end);
                String rest = reply.substring(restStart, toEnd ? reply.length() : end).strip();
                return rest.isEmpty() ? Optional.empty() : Optional.of(rest);
            }
            start = end + 1;
        }
        return Optional.empty();
    }

    /** Skips leading spaces, Markdown and punctuation (anything but a letter or digit) within one line. */
    private static int skipDecoration(String s, int from, int lineEnd) {
        int i = from;
        while (i < lineEnd && !Character.isLetterOrDigit(s.charAt(i))) {
            i++;
        }
        return i;
    }

    /** After the label, skips only spaces and Markdown emphasis, so an answer like "-5" or "(a)" keeps its first char. */
    private static int skipEmphasis(String s, int from, int lineEnd) {
        int i = from;
        while (i < lineEnd && (Character.isWhitespace(s.charAt(i)) || s.charAt(i) == '*' || s.charAt(i) == '_')) {
            i++;
        }
        return i;
    }
}
