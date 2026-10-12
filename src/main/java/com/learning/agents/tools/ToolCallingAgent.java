package com.learning.agents.tools;

import com.learning.agents.OllamaConfig;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.SystemMessage;

/**
 * A LangChain4j "AI Service" agent: the model reasons over the question and calls
 * {@link Calculator} tools as needed. Requires an Ollama model with tool-calling support.
 */
public class ToolCallingAgent {

    interface MathAssistant {
        @SystemMessage("You are a precise math assistant. Always use the provided tools for arithmetic.")
        String chat(String userMessage);
    }

    public static void main(String[] args) {
        MathAssistant assistant = AiServices.builder(MathAssistant.class)
                .chatModel(OllamaConfig.chatModel())
                .tools(new Calculator())
                .chatMemory(MessageWindowChatMemory.withMaxMessages(20))
                .build();

        String question = args.length > 0 ? String.join(" ", args)
                : "What is the square root of (12.5 multiplied by 8) plus 4?";

        System.out.println("Question: " + question);
        System.out.println("Answer  : " + assistant.chat(question));
    }
}
