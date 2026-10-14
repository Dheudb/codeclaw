import type { EngineEvent, InitialSession, SessionView } from "./types.ts";

export function createSession(initial: InitialSession): SessionView {
  return {
    ...initial,
    permissionMode: initial.permissionMode || "ask",
    busy: false,
    error: "",
    messages: initial.history,
    streamIndex: null,
    tools: initial.activity || [],
    workers: initial.workers || [],
    interactions: [],
    loaded: [],
    logs: [],
    selected: [],
    mode: "normal",
    draft: initial.savedDraft || "",
  };
}

export function beginTurn(session: SessionView, text: string): SessionView {
  return {
    ...session,
    busy: true,
    error: "",
    draft: "",
    streamIndex: session.messages.length + 1,
    messages: [
      ...session.messages,
      { role: "user", text },
      { role: "assistant", text: "", trace: { id: "pending", startedAt: Date.now(), status: "running", steps: [] } },
    ],
  };
}

export function applyEvent(
  session: SessionView,
  event: EngineEvent,
): SessionView {
  if (event.session !== session.id) return session;
  switch (event.type) {
    case "file_changed":
      return { ...session, fileChanges: [event.change, ...(session.fileChanges || []).filter((file) => file.path !== event.change.path)].slice(0, 100) };
    case "permission_mode_changed":
      return ["ask", "full"].includes(event.permissionMode) ? { ...session, permissionMode: event.permissionMode } : session;
    case "session_started":
      return {
        ...session,
        persistedId: event.sessionId,
        config: { ...session.config, model: event.model },
      };
    case "turn_trace":
      return session.streamIndex === null ? session : { ...session, messages: session.messages.map((message, index) =>
        index === session.streamIndex ? { ...message, trace: event.trace, text: event.trace.answer || event.trace.pendingText || "" } : message) };
    case "text_delta":
    case "final": {
      if (session.streamIndex === null) return session;
      const messages = session.messages.map((message, index) =>
        index === session.streamIndex
          ? {
              ...message,
              text:
                event.type === "final"
                  ? event.text || message.text
                  : message.text + (event.text || ""),
            }
          : message,
      );
      return { ...session, messages };
    }
    case "turn_finished":
      return { ...session, busy: false, streamIndex: null, messages: finishTrace(session, "completed") };
    case "turn_cancelled":
      return {
        ...session,
        busy: false,
        messages: finishTrace(session, "cancelled"),
        error: "任务已停止，已收到的内容保留。",
        interactions: [],
        tools: settleTools(session, "cancelled"),
      };
    case "turn_error":
    case "engine_exit":
      return {
        ...session,
        busy: false,
        streamIndex: null,
        messages: finishTrace(session, event.type === "engine_exit" ? "interrupted" : "error").filter((message, index) =>
          index !== session.streamIndex || !!message.text.trim() || !!message.trace?.steps.length || !!message.trace && message.trace.id !== "pending"),
        error: (event.message || "引擎发生错误").replace(/\[\/?bold red\]/g, ""),
        interactions: [],
        tools: settleTools(session, "interrupted", event.type === "engine_exit"),
      };
    case "permission":
    case "questions":
    case "plan_approval":
      return { ...session, interactions: [...session.interactions, event],
        tools: event.type === "permission" ? session.tools.map((tool) =>
          tool.id === `${event.request?.classifier_input?.agent_id && event.request.classifier_input.agent_role !== "main" ? event.request.classifier_input.agent_id : "main"}:${event.request?.classifier_input?.tool_use_id}`
            ? { ...tool, status: "waiting", waitingRequestId: event.requestId } : tool) : session.tools };
    case "interaction_resolved":
      return {
        ...session,
        interactions: session.interactions.filter(
          (item) => item.requestId !== event.requestId,
        ),
        tools: session.tools.map((tool) => tool.status === "waiting" && tool.waitingRequestId === event.requestId ? { ...tool, status: "running", waitingRequestId: undefined } : tool),
      };
    case "skill_loaded":
      return {
        ...session,
        loaded: [...new Set([...session.loaded, event.name])],
      };
    case "subagent_status": {
      const previous = session.workers.find(
        (item) => item.agent_id === event.agent.agent_id,
      );
      const worker = { ...previous, ...event.agent };
      const tools = upsertTool(session.tools, { id: "agent:" + worker.agent_id, name: "subagent",
        status: worker.status, agentId: worker.agent_id, agentRole: worker.agent_role || worker.agent_type,
        input: { description: worker.description, prompt: worker.task }, result: worker.result,
        startedAt: worker.started_at || session.tools.find((tool) => tool.id === "agent:" + worker.agent_id)?.startedAt || Date.now(),
        ...(worker.status !== "running" ? { finishedAt: worker.finished_at || Date.now() } : {}) });
      return {
        ...session,
        workers: previous ? session.workers.map((item) => item.agent_id === worker.agent_id ? worker : item) : [...session.workers, worker],
        tools: tools.map((tool) => tool.agentId === worker.agent_id && worker.status !== "running" && ["running", "waiting", "queued"].includes(tool.status)
          ? { ...tool, status: worker.status === "cancelled" ? "cancelled" : "interrupted", finishedAt: Date.now() } : tool),
      };
    }
    case "tool_scheduled":
    case "tool_started": {
      const id = `${event.agent_id || "main"}:${event.tool_use_id || event.id}`;
      const previous = session.tools.find((tool) => tool.id === id);
      return { ...session, tools: upsertTool(session.tools, { id, name: event.tool_name || event.name || "tool",
        status: event.type === "tool_scheduled" ? previous?.status || "queued" : "running",
        agentId: event.agent_id, agentRole: event.agent_role,
        ...(event.input ? { input: event.input } : {}),
        startedAt: previous?.startedAt || event.timestamp || Date.now() }) };
    }
    case "tool_result":
      return { ...session, tools: upsertTool(session.tools, {
        id: `${event.agent_id || "main"}:${event.tool_use_id}`, name: event.tool_name || session.tools.find((tool) => tool.id === `${event.agent_id || "main"}:${event.tool_use_id}`)?.name || "tool",
        agentId: event.agent_id, agentRole: event.agent_role,
        status: event.status === "cancelled" ? "cancelled" : event.is_error ? "error" : "completed", finishedAt: event.timestamp || Date.now(),
        result: typeof (event.result ?? event.content) === "string" ? event.result ?? event.content : JSON.stringify(event.result ?? event.content ?? ""),
      }) };
    case "log":
      return {
        ...session,
        logs: [...session.logs, String(event.text).replace(/\[\/?(?:bold|dim|red|green|yellow|cyan|magenta|blue)(?:[ a-z]*)\]/g, "")].slice(-200),
      };
    case "sessions_changed":
      return { ...session, sessions: event.sessions };
    default:
      return session;
  }
}

function upsertTool(tools: SessionView["tools"], value: SessionView["tools"][number]) {
  const previous = tools.find((tool) => tool.id === value.id);
  return (previous ? tools.map((tool) => tool.id === value.id ? { ...tool, ...value } : tool) : [...tools, value]).slice(-200);
}
function finishTrace(session: SessionView, status: string) {
  return session.messages.map((message, index) => index === session.streamIndex && message.trace?.status === "running"
    ? { ...message, trace: { ...message.trace, status, finishedAt: Date.now(), elapsedMs: Date.now() - (message.trace.startedAt || Date.now()) } } : message);
}
function settleTools(session: SessionView, status: string, includeChildren = true) {
  return session.tools.map((tool) => (includeChildren || !tool.agentId) && ["running", "queued", "waiting"].includes(tool.status)
    ? { ...tool, status, finishedAt: Date.now() } : tool);
}
