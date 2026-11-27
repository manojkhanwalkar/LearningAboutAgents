package com.learning.agents.llm;

import java.util.List;

/**
 * Parses the lists models write ("1. step", "- step", "* step", "• step") into clean items. Used wherever a sample
 * asks the model for a plan or a list of subtasks.
 */
public final class TextLists {

    private TextLists() {
    }

    /**
     * Returns the non-blank lines of {@code text} with list markers removed, skipping preamble lines that end with a
     * colon ("Here is the plan:"), and keeping at most {@code limit} items.
     */
    public static List<String> items(String text, int limit) {
        return text.lines()
                .map(line -> line.strip().replaceFirst("^([-*•]|\\d+[.)])\\s*", ""))
                .filter(line -> !line.isBlank() && !line.endsWith(":"))
                .limit(limit)
                .toList();
    }
}
