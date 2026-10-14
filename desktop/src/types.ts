export type Skill = {
  id: string;
  name: string;
  description: string;
  root: string;
  path: string;
  scope: string;
  enabled: boolean;
  valid: boolean;
  error: string;
  shadowed: boolean;
  compatibility: string;
};
export type AgentDefinition = {
  name: string;
  description?: string;
  when_to_use?: string;
  scope: string;
  valid: boolean;
  shadowed: boolean;
  error?: string;
  model?: string;
  allowed_tools?: string[];
  skills?: string[];
  path?: string;
};
export type Capabilities = {
  skills: Skill[];
  agents: AgentDefinition[];
  settings: {
    skill_roots?: string[];
    agent_roots?: string[];
    disabled_skills?: string[];
    projectSkillsRoot?: string;
    projectAgentsRoot?: string;
    globalSkillsRoot?: string;
    globalAgentsRoot?: string;
  };
};
export type ModelConfig = {
  provider: string;
  model: string;
  baseUrl: string;
  hasKey: boolean;
  networkMode?: "direct" | "system";
};
export type HistoryItem = { role: "user" | "assistant"; text: string };
export type SavedSession = { id: string; title: string; updated: number };
export type CatalogSession = SavedSession & {
  sessionId: string;
  project: string;
  created: number;
  pinned: boolean;
  archived: boolean;
  error?: string;
  draft?: string;
};
export type CatalogProject = {
  path: string;
  name: string;
  open: boolean;
  pinned: boolean;
  archived: boolean;
  collapsed?: boolean;
  error?: string;
  sessions: CatalogSession[];
};
export type SessionCatalog = {
  revision?: number;
  projects: CatalogProject[];
  view: { sort: "recent" | "oldest" | "name" | "created"; showArchived: boolean };
};
export type InitialSession = {
  id: string;
  persistedId?: string;
  project: string;
  title?: string;
  savedDraft?: string;
  config: ModelConfig;
  capabilities: Capabilities;
  sessions: SavedSession[];
  history: HistoryItem[];
};
export type EngineEvent = { session: string; type: string; [key: string]: any };
export type Worker = {
  agent_id: string;
  agent_type?: string;
  status: string;
  task?: string;
  result?: string;
  model?: string;
  background?: boolean;
};
export type ToolActivity = {
  id: string;
  name: string;
  status: string;
  result?: string;
};
export type SessionView = InitialSession & {
  persistedId?: string;
  busy: boolean;
  error: string;
  messages: HistoryItem[];
  streamIndex: number | null;
  tools: ToolActivity[];
  workers: Worker[];
  interactions: EngineEvent[];
  loaded: string[];
  logs: string[];
  selected: string[];
  mode: string;
  draft: string;
};
declare global {
  interface Window {
    codeclaw: {
      invoke<T = any>(
        method: string,
        params?: Record<string, unknown>,
      ): Promise<T>;
      onEvent(callback: (event: EngineEvent) => void): () => void;
    };
  }
}
