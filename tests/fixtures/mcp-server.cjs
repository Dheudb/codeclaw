// Local-only protocol fixture: no external API, model, or paid service.
const http = require("node:http");
const readline = require("node:readline");
function respond(request, authorization = "") {
  if (request.id === undefined) return null;
  let result;
  if (request.method === "initialize")
    result = {
      protocolVersion: request.params.protocolVersion,
      capabilities: { tools: {} },
      serverInfo: { name: "codeclaw-search-fixture", version: "1.0" },
    };
  else if (request.method === "ping") result = {};
  else if (request.method === "tools/list")
    result = {
      tools: [
        {
          name: "search",
          description: "Search fixture for offline acceptance",
          inputSchema: {
            type: "object",
            properties: { query: { type: "string" } },
            required: ["query"],
          },
        },
      ],
    };
  else if (request.method === "tools/call")
    result = {
      content: [
        {
          type: "text",
          text:
            request.params.arguments.query === "__header__"
              ? authorization
              : request.params.arguments.query === "__env__"
                ? (process.env.CODECLAW_TEST_MODEL_SECRET || "absent") +
                  ":" +
                  process.env.SEARCH_API_KEY
                : "SEARCH_FIXTURE_PROOF: " +
                  request.params.arguments.query +
                  " https://example.invalid/source",
        },
      ],
      isError: request.params.arguments.query === "fail",
    };
  else
    return {
      jsonrpc: "2.0",
      id: request.id,
      error: { code: -32601, message: "Unknown method" },
    };
  return { jsonrpc: "2.0", id: request.id, result };
}
if (process.argv.includes("--http")) {
  const server = http.createServer((request, response) => {
    if (request.method !== "POST") {
      response.writeHead(request.method === "DELETE" ? 200 : 405).end();
      return;
    }
    let data = "";
    request.on("data", (chunk) => {
      data += chunk;
    });
    request.on("end", () => {
      const result = respond(JSON.parse(data), request.headers.authorization);
      if (!result) response.writeHead(202).end();
      else
        response
          .writeHead(200, { "Content-Type": "application/json" })
          .end(JSON.stringify(result));
    });
  });
  server.listen(0, "127.0.0.1", () =>
    process.stdout.write(
      JSON.stringify({ port: server.address().port }) + "\n",
    ),
  );
} else {
  readline.createInterface({ input: process.stdin }).on("line", (line) => {
    const result = respond(JSON.parse(line));
    if (result) process.stdout.write(JSON.stringify(result) + "\n");
  });
}
