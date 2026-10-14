const test = require("node:test");
const assert = require("node:assert/strict");
const state = import("../src/state.ts");
const initial = {
  id: "one",
  project: "/project",
  config: { provider: "openai", model: "test", baseUrl: "", hasKey: false },
  capabilities: { skills: [], agents: [], settings: {} },
  sessions: [],
  history: [],
};
test("a failed request removes its empty assistant placeholder and Rich tags", async () => {
  const { createSession, beginTurn, applyEvent } = await state;
  const session = applyEvent(beginTurn(createSession(initial), "hello"), {
    session: "one", type: "turn_error", message: "[bold red]API Loop Exception:[/bold red] Connection error.",
  });
  assert.deepEqual(session.messages, [{ role: "user", text: "hello" }]);
  assert.equal(session.error, "API Loop Exception: Connection error.");
});
test("events cannot alter another project session", async () => {
  const { createSession, applyEvent } = await state;
  const session = createSession(initial);
  assert.equal(
    applyEvent(session, {
      session: "two",
      type: "turn_error",
      message: "wrong session",
    }),
    session,
  );
});
test("streaming and final answer share one assistant message", async () => {
  const { createSession, beginTurn, applyEvent } = await state;
  let session = beginTurn(createSession(initial), "hello");
  session = applyEvent(session, {
    session: "one",
    type: "text_delta",
    text: "part",
  });
  session = applyEvent(session, {
    session: "one",
    type: "final",
    text: "complete",
  });
  session = applyEvent(session, { session: "one", type: "turn_finished" });
  assert.equal(session.messages.length, 2);
  assert.equal(session.messages[1].text, "complete");
  assert.equal(session.busy, false);
});
test("parallel approvals resolve individually", async () => {
  const { createSession, applyEvent } = await state;
  let session = createSession(initial);
  session = applyEvent(session, {
    session: "one",
    type: "permission",
    requestId: "a",
  });
  session = applyEvent(session, {
    session: "one",
    type: "permission",
    requestId: "b",
  });
  session = applyEvent(session, {
    session: "one",
    type: "interaction_resolved",
    requestId: "a",
  });
  assert.deepEqual(
    session.interactions.map((item) => item.requestId),
    ["b"],
  );
});
test("worker completion preserves task identity and model", async () => {
  const { createSession, applyEvent } = await state;
  let session = createSession(initial);
  session = applyEvent(session, {
    session: "one",
    type: "subagent_status",
    agent: {
      agent_id: "a",
      status: "running",
      model: "review",
      task: "inspect",
    },
  });
  session = applyEvent(session, {
    session: "one",
    type: "subagent_status",
    agent: { agent_id: "a", status: "completed", result: "done" },
  });
  assert.equal(session.workers.length, 1);
  assert.equal(session.workers[0].model, "review");
  assert.equal(session.workers[0].result, "done");
});
