package com.learning.agents.tools.mcp;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.InputStream;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.util.Locale;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * A deliberately tiny MCP (Model Context Protocol) server, so the MCP sample can run without installing anything.
 * It speaks the Streamable HTTP transport: the client POSTs one JSON-RPC message at a time and gets a JSON reply.
 * It supports {@code initialize}, {@code ping}, {@code tools/list} and {@code tools/call}, and offers two tools,
 * {@code capital_of} and {@code celsius_to_fahrenheit}.
 * <p>
 * It is a teaching aid, not a complete server: no SSE stream, no resources or prompts, no auth. Its protections are
 * the ones the MCP spec asks every local server for: it binds to the loopback address only, on a free port; it rejects
 * requests whose {@code Origin} is not this server (403), which stops a web page from reaching it, including through
 * DNS rebinding; it accepts only {@code Content-Type: application/json} (415); and it caps request bodies at
 * {@link #MAX_BODY_BYTES} (413). Real MCP servers are usually built with an MCP SDK. Close it to stop it.
 */
public final class DemoMcpServer implements AutoCloseable {

    static final int MAX_BODY_BYTES = 64 * 1024;
    static final String PROTOCOL_VERSION = "2025-06-18";

    private static final ObjectMapper JSON = new ObjectMapper();
    private static final Map<String, String> CAPITALS = Map.of(
            "france", "Paris", "japan", "Tokyo", "india", "New Delhi", "kenya", "Nairobi", "peru", "Lima");

    private final HttpServer http;
    private final ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor();
    private final String sessionId = UUID.randomUUID().toString();

    private DemoMcpServer(HttpServer http) {
        this.http = http;
    }

    /** Starts the server on a free loopback port. */
    public static DemoMcpServer start() throws IOException {
        // 127.0.0.1 explicitly, not getLoopbackAddress(), which is ::1 when the JVM prefers IPv6 and wouldn't match url().
        var http = HttpServer.create(new InetSocketAddress(InetAddress.getByName("127.0.0.1"), 0), 0);
        var server = new DemoMcpServer(http);
        http.createContext("/mcp", server::handle);
        http.setExecutor(server.executor);
        http.start();
        return server;
    }

    /** The URL to give an MCP client. */
    public String url() {
        return "http://127.0.0.1:" + http.getAddress().getPort() + "/mcp";
    }

    @Override
    public void close() {
        http.stop(0);
        executor.close();
    }

    private void handle(HttpExchange exchange) throws IOException {
        try (exchange) {
            if (!"POST".equals(exchange.getRequestMethod())) {
                // No server-to-client SSE stream and no session deletion in this tiny server.
                exchange.sendResponseHeaders(405, -1);
                return;
            }
            String origin = exchange.getRequestHeaders().getFirst("Origin");
            if (origin != null && !isOwnOrigin(origin)) {
                exchange.sendResponseHeaders(403, -1);   // a browser page from another site
                return;
            }
            String contentType = exchange.getRequestHeaders().getFirst("Content-Type");
            if (contentType == null || !contentType.toLowerCase(Locale.ROOT).startsWith("application/json")) {
                exchange.sendResponseHeaders(415, -1);
                return;
            }
            byte[] body = readLimited(exchange.getRequestBody());
            if (body == null) {
                exchange.sendResponseHeaders(413, -1);
                return;
            }
            JsonNode request;
            try {
                request = JSON.readTree(body);
            } catch (IOException _) {
                send(exchange, 400, error(null, -32700, "parse error"));
                return;
            }
            if (request == null || !request.isObject()) {
                // Empty bodies, bare values and JSON-RPC batches (arrays) aren't supported by this tiny server.
                send(exchange, 400, error(null, -32600, "invalid request: expected one JSON-RPC object"));
                return;
            }
            if (!request.has("id")) {
                if (!request.has("method")) {
                    send(exchange, 400, error(null, -32600, "invalid request: no method"));
                    return;
                }
                exchange.sendResponseHeaders(202, -1);   // a notification, e.g. notifications/initialized
                return;
            }
            send(exchange, 200, respond(request));
        }
    }

    /** True only for this server's own origin, under either loopback name. */
    boolean isOwnOrigin(String origin) {
        int port = http.getAddress().getPort();
        return origin.equals("http://127.0.0.1:" + port) || origin.equals("http://localhost:" + port);
    }

    /** Answers one JSON-RPC request; package-private so the protocol can be unit-tested without HTTP. */
    static ObjectNode respond(JsonNode request) {
        JsonNode id = request.get("id");
        String method = request.path("method").asText();
        return switch (method) {
            case "initialize" -> result(id, JSON.createObjectNode()
                    .put("protocolVersion", PROTOCOL_VERSION)
                    .<ObjectNode>set("capabilities", JSON.createObjectNode().set("tools", JSON.createObjectNode()))
                    .set("serverInfo", JSON.createObjectNode().put("name", "demo-mcp-server").put("version", "1.0")));
            case "ping" -> result(id, JSON.createObjectNode());
            case "tools/list" -> result(id, toolList());
            case "tools/call" -> result(id, callTool(request.path("params")));
            default -> error(id, -32601, "method not found: " + method);
        };
    }

    private static ObjectNode toolList() {
        var tools = JSON.createArrayNode();
        tools.add(tool("capital_of", "Returns the capital city of a country.", "country", "string", "the country, e.g. Japan"));
        tools.add(tool("celsius_to_fahrenheit", "Converts a temperature from Celsius to Fahrenheit.",
                "celsius", "number", "the temperature in degrees Celsius"));
        return JSON.createObjectNode().set("tools", tools);
    }

    private static ObjectNode tool(String name, String description, String param, String type, String paramDescription) {
        var schema = JSON.createObjectNode().put("type", "object");
        schema.putObject("properties").putObject(param).put("type", type).put("description", paramDescription);
        schema.putArray("required").add(param);
        return JSON.createObjectNode().put("name", name).put("description", description).set("inputSchema", schema);
    }

    private static ObjectNode callTool(JsonNode params) {
        String name = params.path("name").asText();
        JsonNode args = params.path("arguments");
        return switch (name) {
            case "capital_of" -> {
                String country = args.path("country").asText("").strip().toLowerCase(Locale.ROOT);
                String capital = CAPITALS.get(country);
                yield capital == null ? toolText("No capital known for '" + country + "'.", true) : toolText(capital, false);
            }
            case "celsius_to_fahrenheit" -> args.path("celsius").isNumber()
                    ? toolText(String.format(Locale.ROOT, "%.1f", args.path("celsius").asDouble() * 9 / 5 + 32), false)
                    : toolText("'celsius' must be a number.", true);
            default -> toolText("Unknown tool: " + name, true);
        };
    }

    private static ObjectNode toolText(String text, boolean isError) {
        var result = JSON.createObjectNode().put("isError", isError);
        result.putArray("content").addObject().put("type", "text").put("text", text);
        return result;
    }

    private static ObjectNode result(JsonNode id, JsonNode result) {
        ObjectNode node = JSON.createObjectNode().put("jsonrpc", "2.0");
        node.set("id", id);
        node.set("result", result);
        return node;
    }

    private static ObjectNode error(JsonNode id, int code, String message) {
        ObjectNode node = JSON.createObjectNode().put("jsonrpc", "2.0");
        node.set("id", id);
        node.putObject("error").put("code", code).put("message", message);
        return node;
    }

    private void send(HttpExchange exchange, int status, ObjectNode body) throws IOException {
        byte[] bytes = JSON.writeValueAsBytes(body);
        exchange.getResponseHeaders().set("Content-Type", "application/json");
        exchange.getResponseHeaders().set("Mcp-Session-Id", sessionId);
        exchange.sendResponseHeaders(status, bytes.length);
        exchange.getResponseBody().write(bytes);
    }

    /** Reads at most {@link #MAX_BODY_BYTES}; returns null if the body is larger. */
    private static byte[] readLimited(InputStream in) throws IOException {
        byte[] bytes = in.readNBytes(MAX_BODY_BYTES + 1);
        return bytes.length > MAX_BODY_BYTES ? null : bytes;
    }

    /** For trying the server with another MCP client: prints the URL and serves until the process is stopped. */
    public static void main(String[] args) throws Exception {
        try (var server = start()) {
            System.out.println("Demo MCP server listening on " + server.url() + " (Ctrl-C to stop)");
            Thread.currentThread().join();
        }
    }
}
