import type { EngineEvent, InitialSession, SessionView } from "./types.ts";

export function createSession(initial: InitialSession): SessionView {
  return {
    ...initial,
    busy: false,
    error: "",
    messages: initial.history,
    streamIndex: null,
    tools: [],
    workers: [],
    interactions: [],
    loaded: [],
    logs: [],
    selected: [],
    mode: "normal",
    draft: "",
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
      { role: "assistant", text: "" },
    ],
  };
}

export function applyEvent(
  session: SessionView,
  event: EngineEvent,
): SessionView {
  if (event.session !== session.id) return session;
  switch (event.type) {
    case "session_started":
      return {
        ...session,
        persistedId: event.sessionId,
        config: { ...session.config, model: event.model },
      };
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
      return { ...session, busy: false, streamIndex: null };
    case "turn_cancelled":
      return {
        ...session,
        busy: false,
        error: "任务已停止，已收到的内容保留。",
        interactions: [],
      };
    case "turn_error":
    case "engine_exit":
      return {
        ...session,
        busy: false,
        streamIndex: null,
        error: event.message || "引擎发生错误",
        interactions: [],
      };
    case "permission":
    case "questions":
    case "plan_approval":
      return { ...session, interactions: [...session.interactions, event] };
    case "interaction_resolved":
      return {
        ...session,
        interactions: session.interactions.filter(
          (item) => item.requestId !== event.requestId,
        ),
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
      return {
        ...session,
        workers: [
          ...session.workers.filter(
            (item) => item.agent_id !== event.agent.agent_id,
          ),
          { ...previous, ...event.agent },
        ],
      };
    }
    case "tool_scheduled":
      return {
        ...session,
        tools: [
          ...session.tools,
          {
            id: event.tool_use_id || event.id || String(session.tools.length),
            name: event.tool_name || event.name || "tool",
            status: "running",
          },
        ].slice(-80),
      };
    case "tool_result":
      return {
        ...session,
        tools: session.tools.map((tool) =>
          tool.id === event.tool_use_id
            ? {
                ...tool,
                status: event.is_error ? "error" : "completed",
                result:
                  typeof event.result === "string"
                    ? event.result
                    : JSON.stringify(event.result || event.content || ""),
              }
            : tool,
        ),
      };
    case "log":
      return {
        ...session,
        logs: [...session.logs, String(event.text)].slice(-80),
      };
    case "sessions_changed":
      return { ...session, sessions: event.sessions };
    default:
      return session;
  }
}
