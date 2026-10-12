package com.learning.agents.tools;

import dev.langchain4j.agent.tool.P;
import dev.langchain4j.agent.tool.Tool;

/**
 * Tools the LLM may decide to call. Each {@link Tool} description is sent to the model.
 */
public class Calculator {

    @Tool("Adds two numbers")
    public double add(@P("first number") double a, @P("second number") double b) {
        System.out.printf("  [tool] add(%s, %s)%n", a, b);
        return a + b;
    }

    @Tool("Multiplies two numbers")
    public double multiply(@P("first number") double a, @P("second number") double b) {
        System.out.printf("  [tool] multiply(%s, %s)%n", a, b);
        return a * b;
    }

    @Tool("Returns the square root of a number")
    public double sqrt(@P("the number") double x) {
        System.out.printf("  [tool] sqrt(%s)%n", x);
        return Math.sqrt(x);
    }
}
