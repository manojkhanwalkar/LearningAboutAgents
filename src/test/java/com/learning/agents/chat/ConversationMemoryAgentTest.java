package com.learning.agents.chat;

import com.learning.agents.MainRunner;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ConversationMemoryAgentTest {

    @Test
    void recentTurnsAreResentVerbatim() {
        List<String> prompts = new ArrayList<>();
        var agent = new ConversationMemoryAgent(prompt -> {
            prompts.add(prompt);
            return "ok";
        });

        agent.chat("I like tea");
        agent.chat("and you?");

        assertTrue(prompts.get(1).contains("User: I like tea\nAssistant: ok"));
        assertTrue(prompts.get(1).contains("Summary of earlier conversation: (none)"));
    }

    @Test
    void oldTurnsAreFoldedIntoTheSummaryAndMemoryStaysBounded() {
        List<String> summaryPrompts = new ArrayList<>();
        var agent = new ConversationMemoryAgent(prompt -> {
            if (prompt.startsWith("Update the running summary")) {
                summaryPrompts.add(prompt);
                return "summary #" + summaryPrompts.size();
            }
            return "ok";
        });

        for (int i = 1; i <= 6; i++) {
            agent.chat("message " + i);
            assertTrue(agent.recent().size() <= ConversationMemoryAgent.RECENT_TURNS);
        }

        assertFalse(summaryPrompts.isEmpty());
        assertTrue(summaryPrompts.getFirst().contains("User: message 1"));
        assertTrue(summaryPrompts.get(1).contains("Current summary: summary #1"));
        assertEquals("summary #" + summaryPrompts.size(), agent.summary());
        assertTrue(agent.prompt("next").contains("Summary of earlier conversation: summary #"));
        assertFalse(agent.prompt("next").contains("message 1"));
    }

    @Test
    void placeholderOnlyRemembersTheNameThroughTheSummary() {
        var agent = new ConversationMemoryAgent(ConversationMemoryAgent.placeholder());
        assertEquals("Sorry, I don't remember.", agent.chat("What's my name?"));
    }

    @Test
    void mainRecallsTheNameFromTheSummary() throws Exception {
        assertTrue(MainRunner.run(ConversationMemoryAgent::main).contains("Your name is Priya."));
        assertTrue(MainRunner.run(ConversationMemoryAgent::main, "hello").contains("User     : hello"));
    }
}
