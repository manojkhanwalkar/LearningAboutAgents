package com.learning.agents.chat;

import com.learning.agents.OllamaConfig;
import dev.langchain4j.model.chat.ChatModel;

/**
 * Smallest possible LangChain4j example: send one prompt to Ollama and print the reply.
 */
public class HelloOllama {

    public static void main(String[] args) {
        ChatModel model = OllamaConfig.chatModel();
        String prompt = args.length > 0 ? String.join(" ", args) : "In one sentence, what is an AI agent?";

        System.out.println("Model : " + OllamaConfig.modelName());
        System.out.println("Prompt: " + prompt);
        System.out.println("Reply : " + model.chat(prompt));
    }
}
