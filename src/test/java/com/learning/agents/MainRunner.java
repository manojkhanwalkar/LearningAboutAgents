package com.learning.agents;

import com.learning.agents.llm.LlmProvider;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Runs a sample's {@code main} against its scripted placeholder LLM and returns what it printed.
 * Skips the test if {@code LLM_BACKEND} is set, so the test suite never calls a real model.
 */
public final class MainRunner {

    /** A {@code main} method reference. */
    @FunctionalInterface
    public interface Main {
        void run(String[] args) throws Exception;
    }

    private MainRunner() {
    }

    public static String run(Main main, String... args) throws Exception {
        return runWithInput(main, "", args);
    }

    /** Like {@link #run}, with {@code stdin} as the program's standard input. */
    public static String runWithInput(Main main, String stdin, String... args) throws Exception {
        String backend = System.getenv(LlmProvider.ENV);
        assumeTrue(backend == null || backend.isBlank(), "LLM_BACKEND is set; skipping the placeholder run");

        PrintStream originalOut = System.out;
        InputStream originalIn = System.in;
        var buffer = new ByteArrayOutputStream();
        try (var out = new PrintStream(buffer, true, StandardCharsets.UTF_8)) {
            System.setOut(out);
            System.setIn(new ByteArrayInputStream(stdin.getBytes(StandardCharsets.UTF_8)));
            main.run(args);
        } finally {
            System.setOut(originalOut);
            System.setIn(originalIn);
        }
        return buffer.toString(StandardCharsets.UTF_8);
    }
}
