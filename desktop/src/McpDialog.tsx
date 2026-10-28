import { useEffect, useState } from "react";
import {
  X,
  Search,
  Plus,
  Plug,
  Globe,
  Terminal,
  LoaderCircle,
  RefreshCw,
  Check,
  ExternalLink,
} from "lucide-react";

type Config = {
  type: string;
  command?: string;
  args?: string[];
  url?: string;
  env?: Record<string, string>;
  headers?: Record<string, string>;
  enabled?: boolean;
  label?: string;
  catalogId?: string;
  savedUrlQuery?: boolean;
};
type Entry = {
  id: string;
  name: string;
  description: string;
  source: string;
  homepage: string;
  requirements: string;
  config: Config;
  inputs: {
    name: string;
    description: string;
    optional?: boolean;
    secret?: boolean;
    defaultValue?: string;
  }[];
};
type Server = Config & {
  name: string;
  error?: string;
  status: string;
  connection: { status?: string; error?: string; tools?: string[] };
};
type Snapshot = {
  servers: Server[];
  runtime: {
    available: boolean;
    python: string;
    executable: string;
    message: string;
  };
  needsReload: boolean;
  project: string;
};
type Editor = {
  entry?: Entry;
  server?: Server;
  name: string;
  type: string;
  command: string;
  args: string;
  url: string;
  pairs: { key: string; value: string }[];
  values: Record<string, string>;
};
const statusText: Record<string, string> = {
  connected: "已连接",
  disabled: "已停用",
  pending: "下次任务连接",
  error: "连接失败",
};

export default function McpDialog({
  sessionId,
  busy,
  invoke,
  close,
}: {
  sessionId: string;
  busy: boolean;
  invoke: (method: string, params?: Record<string, unknown>) => Promise<any>;
  close: () => void;
}) {
  const [tab, setTab] = useState("market");
  const [source, setSource] = useState("builtin");
  const [query, setQuery] = useState("");
  const [entries, setEntries] = useState<Entry[]>([]);
  const [cursor, setCursor] = useState("");
  const [skipped, setSkipped] = useState(0);
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [editor, setEditor] = useState<Editor | null>(null);
  const [loading, setLoading] = useState(false);
  const [working, setWorking] = useState("");
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [tests, setTests] = useState<
    Record<string, { status: string; error?: string; tools?: string[] }>
  >({});
  const [removing, setRemoving] = useState("");
  const locked = busy || !!working;

  useEffect(() => {
    let alive = true;
    invoke("mcp.list")
      .then((value) => alive && setSnapshot(value))
      .catch((e) => alive && setError(String(e)));
    load("builtin", "");
    const unsubscribe = window.xuloom.onEvent((event) => {
      if (event.session === sessionId && event.type === "mcp_changed")
        setSnapshot(event.snapshot);
    });
    return () => {
      alive = false;
      unsubscribe();
    };
  }, [sessionId]);

  async function load(nextSource = source, nextQuery = query, nextCursor = "") {
    setLoading(true);
    setError("");
    try {
      const result = await invoke("mcp.market", {
        source: nextSource,
        query: nextQuery,
        cursor: nextCursor,
      });
      setEntries((previous) =>
        nextCursor
          ? [
              ...previous,
              ...result.entries.filter(
                (item: Entry) => !previous.some((old) => old.id === item.id),
              ),
            ]
          : result.entries,
      );
      setCursor(result.cursor || "");
      setSkipped(result.skipped || 0);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  function edit(entry?: Entry, server?: Server) {
    const config = server || entry?.config;
    setRemoving("");
    setError("");
    setMessage("");
    setEditor({
      entry,
      server,
      name: server?.name || entry?.id || "",
      type: config?.type || "http",
      command: config?.command || "",
      args: JSON.stringify(config?.args || []),
      url: config?.url || "",
      pairs: Object.entries(
        config?.type === "stdio" ? config.env || {} : config?.headers || {},
      ).map(([key, value]) => ({ key, value })),
      values: Object.fromEntries(
        (entry?.inputs || []).map((spec) => [
          spec.name,
          spec.defaultValue || "",
        ]),
      ),
    });
  }

  async function action(
    method: string,
    params: Record<string, unknown>,
    label: string,
  ) {
    setWorking(label);
    setError("");
    setMessage("");
    try {
      const result = await invoke(method, params);
      if (method === "mcp.test") {
        setTests((old) => ({ ...old, [String(params.name)]: result }));
        setMessage(
          result.status === "connected"
            ? `连接成功，发现 ${result.tools.length} 个工具。`
            : result.error || result.runtime.message,
        );
      } else {
        setSnapshot(result);
        setTests({});
        if (method === "mcp.install" || method === "mcp.save") {
          setEditor(null);
          setTab("installed");
          setMessage("配置已保存。启用后将在下次任务连接；也可以先测试连接。");
        }
      }
      setRemoving("");
    } catch (e) {
      setError(String(e));
    } finally {
      setWorking("");
    }
  }

  async function save() {
    if (!editor) return;
    if (editor.entry)
      return action(
        "mcp.install",
        { id: editor.entry.id, name: editor.name, values: editor.values },
        "save",
      );
    try {
      const mapping = Object.fromEntries(
        editor.pairs
          .filter((p) => p.key.trim())
          .map((p) => [p.key.trim(), p.value]),
      );
      const config: Config = {
        type: editor.type,
        enabled: editor.server?.enabled ?? false,
        label: editor.server?.label,
        catalogId: editor.server?.catalogId,
      };
      if (editor.type === "stdio")
        Object.assign(config, {
          command: editor.command,
          args: JSON.parse(editor.args),
          env: mapping,
        });
      else
        Object.assign(config, {
          url: editor.url,
          headers: mapping,
          savedUrlQuery: editor.server?.savedUrlQuery,
        });
      await action(
        "mcp.save",
        { name: editor.name, config, existing: !!editor.server },
        "save",
      );
    } catch {
      setError('启动参数必须是 JSON 字符串数组，例如 ["-y", "包名"]。');
    }
  }

  return (
    <div className="overlay" onClick={() => !working && close()}>
      <section
        role="dialog"
        aria-modal="true"
        aria-label="MCP 市场"
        className={"dialog mcp-dialog " + (editor ? "with-editor" : "")}
        onClick={(e) => e.stopPropagation()}
      >
        <header>
          <div>
            <span className="eyebrow">TOOLS & CONNECTIONS</span>
            <h2>MCP 市场</h2>
            <p className="mcp-subtitle">为当前项目添加搜索、文档和开发工具</p>
          </div>
          <button
            className="icon-button"
            aria-label="关闭 MCP 市场"
            disabled={!!working}
            onClick={close}
          >
            <X size={18} />
          </button>
        </header>
        <div className="mcp-toolbar">
          <div className="mcp-tabs">
            <button
              className={tab === "market" ? "active" : ""}
              onClick={() => setTab("market")}
            >
              浏览市场
            </button>
            <button
              className={tab === "installed" ? "active" : ""}
              onClick={() => setTab("installed")}
            >
              我的服务 · {snapshot?.servers.length || 0}
            </button>
          </div>
          <button disabled={locked} onClick={() => edit()}>
            <Plus size={14} />
            自定义服务
          </button>
        </div>
        {busy && (
          <p className="notice">
            当前有任务正在运行，完成或停止后可以修改 MCP 配置。
          </p>
        )}
        {snapshot && !snapshot.runtime.available && (
          <p className="mcp-warning">
            {snapshot.runtime.message} 当前 Python：{snapshot.runtime.python}
            。市场浏览和保存配置仍可使用。
          </p>
        )}
        {error && (
          <p role="alert" className="mcp-warning">
            {error.replace(/^Error: /, "")}
          </p>
        )}
        {message && (
          <p role="status" className="mcp-message">
            {message}
          </p>
        )}
        <div className="mcp-body">
          <div className="mcp-main">
            {tab === "market" ? (
              <>
                <form
                  className="mcp-search"
                  onSubmit={(e) => {
                    e.preventDefault();
                    load();
                  }}
                >
                  <select
                    aria-label="市场来源"
                    value={source}
                    disabled={loading}
                    onChange={(e) => {
                      setSource(e.target.value);
                      setEntries([]);
                      setCursor("");
                      load(e.target.value, query);
                    }}
                  >
                    <option value="builtin">精选目录</option>
                    <option value="registry">官方 MCP 注册表</option>
                  </select>
                  <input
                    aria-label="搜索 MCP"
                    placeholder="搜索服务名称或用途…"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                  />
                  <button disabled={loading}>
                    <Search size={15} />
                    {loading ? "搜索中" : "搜索"}
                  </button>
                </form>
                <div className="mcp-grid">
                  {entries.map((entry) => {
                    const installed = snapshot?.servers.some(
                      (s) => s.catalogId === entry.id || s.name === entry.id,
                    );
                    return (
                      <article className="mcp-card" key={entry.id}>
                        <div className="mcp-card-head">
                          {entry.config.type === "http" ? (
                            <Globe size={18} />
                          ) : (
                            <Terminal size={18} />
                          )}
                          <strong>{entry.name}</strong>
                          <span>{entry.source}</span>
                        </div>
                        <p>{entry.description}</p>
                        <small>{entry.requirements}</small>
                        <div className="mcp-card-actions">
                          {entry.homepage && (
                            <button
                              className="mcp-doc-link"
                              onClick={() =>
                                window.xuloom
                                  .invoke("external.open", {
                                    url: entry.homepage,
                                  })
                                  .catch((e) => setError(String(e)))
                              }
                            >
                              <ExternalLink size={12} />
                              文档
                            </button>
                          )}
                          <button
                            disabled={locked || !!installed}
                            onClick={() => edit(entry)}
                          >
                            {installed ? (
                              <>
                                <Check size={13} />
                                已安装
                              </>
                            ) : (
                              <>
                                <Plus size={13} />
                                安装
                              </>
                            )}
                          </button>
                        </div>
                      </article>
                    );
                  })}
                </div>
                {!entries.length && !loading && (
                  <div className="instruction-empty">
                    <Search size={24} />
                    <p>没有匹配的可安装服务。换个关键词或添加自定义服务。</p>
                  </div>
                )}
                {skipped > 0 && (
                  <p className="mcp-footnote">
                    本页 {skipped}{" "}
                    条记录需要暂未支持的启动参数或连接方式，未列入可安装列表。
                  </p>
                )}
                {cursor && (
                  <button
                    disabled={loading}
                    onClick={() => load(source, query, cursor)}
                  >
                    {loading ? "加载中…" : "加载更多"}
                  </button>
                )}
              </>
            ) : (
              <>
                <div className="mcp-installed-heading">
                  <p>配置保存在当前项目，与 CLI 共用。</p>
                  <button
                    disabled={locked}
                    onClick={() => action("mcp.reload", {}, "reload")}
                  >
                    <RefreshCw size={13} />
                    刷新连接
                  </button>
                </div>
                {snapshot?.needsReload && (
                  <p className="notice">
                    配置已改变，下一次任务会重新连接；也可点击刷新连接立即应用到当前会话。
                  </p>
                )}
                {!snapshot?.servers.length && (
                  <div className="instruction-empty">
                    <Plug size={28} />
                    <h4>还没有 MCP 服务</h4>
                    <p>从市场安装搜索服务，或添加自己的连接。</p>
                  </div>
                )}
                {snapshot?.servers.map((server) => {
                  const test = tests[server.name];
                  return (
                    <article className="mcp-card installed" key={server.name}>
                      <div className="mcp-card-head">
                        <Plug size={17} />
                        <strong>{server.label || server.name}</strong>
                        <span className={"mcp-status " + server.status}>
                          {statusText[server.status] || server.status}
                        </span>
                      </div>
                      <code>
                        {server.type === "stdio"
                          ? [server.command, ...(server.args || [])].join(" ")
                          : server.url}
                      </code>
                      <p>
                        {server.error ||
                          server.connection?.error ||
                          test?.error ||
                          (test?.status === "connected"
                            ? `测试成功：${test.tools?.join("、") || "没有工具"}`
                            : server.connection?.tools?.length
                              ? `${server.connection.tools.length} 个工具可用`
                              : server.name)}
                      </p>
                      <div className="mcp-card-actions">
                        <button
                          disabled={locked}
                          onClick={() => edit(undefined, server)}
                        >
                          配置
                        </button>
                        <button
                          disabled={locked || !snapshot.runtime.available}
                          onClick={() =>
                            action(
                              "mcp.test",
                              { name: server.name },
                              server.name,
                            )
                          }
                        >
                          {working === server.name ? (
                            <LoaderCircle className="spin" size={13} />
                          ) : (
                            <Plug size={13} />
                          )}
                          测试连接
                        </button>
                        <button
                          disabled={locked}
                          onClick={() =>
                            action(
                              "mcp.toggle",
                              {
                                name: server.name,
                                enabled: server.enabled === false,
                              },
                              "toggle",
                            )
                          }
                        >
                          {server.enabled === false ? "启用" : "停用"}
                        </button>
                        <button
                          className="danger"
                          disabled={locked}
                          onClick={() =>
                            removing === server.name
                              ? action(
                                  "mcp.remove",
                                  { name: server.name },
                                  "remove",
                                )
                              : setRemoving(server.name)
                          }
                        >
                          {removing === server.name ? "确认移除" : "移除"}
                        </button>
                      </div>
                    </article>
                  );
                })}
              </>
            )}
          </div>
          {editor && (
            <aside className="mcp-editor">
              <div className="mcp-editor-title">
                <h3>
                  {editor.entry
                    ? "安装 " + editor.entry.name
                    : editor.server
                      ? "编辑服务"
                      : "添加自定义 MCP"}
                </h3>
                <button
                  className="icon-button"
                  aria-label="取消 MCP 配置"
                  disabled={!!working}
                  onClick={() => setEditor(null)}
                >
                  <X size={15} />
                </button>
              </div>
              <div className="form-section">
                <label>
                  服务标识
                  <input
                    aria-label="MCP 服务标识"
                    value={editor.name}
                    disabled={!!editor.server || locked}
                    placeholder="my-search"
                    onChange={(e) =>
                      setEditor({ ...editor, name: e.target.value })
                    }
                  />
                </label>
                {editor.entry ? (
                  <>
                    <p className="mcp-footnote">{editor.entry.requirements}</p>
                    <pre>
                      {editor.entry.config.type === "http"
                        ? editor.entry.config.url
                        : [
                            editor.entry.config.command,
                            ...(editor.entry.config.args || []),
                          ].join(" ")}
                    </pre>
                    {editor.entry.inputs.map((spec) => (
                      <label key={spec.name}>
                        {spec.name}
                        {spec.optional ? "（可选）" : " *"}
                        <input
                          type={spec.secret ? "password" : "text"}
                          autoComplete="off"
                          aria-label={spec.name}
                          value={editor.values[spec.name] || ""}
                          onChange={(e) =>
                            setEditor({
                              ...editor,
                              values: {
                                ...editor.values,
                                [spec.name]: e.target.value,
                              },
                            })
                          }
                        />
                        <small>{spec.description}</small>
                      </label>
                    ))}
                  </>
                ) : (
                  <>
                    <label>
                      连接方式
                      <select
                        aria-label="MCP 连接方式"
                        value={editor.type}
                        disabled={locked}
                        onChange={(e) =>
                          setEditor({
                            ...editor,
                            type: e.target.value,
                            pairs: [],
                          })
                        }
                      >
                        <option value="http">
                          远程 HTTP（Streamable HTTP）
                        </option>
                        <option value="stdio">本地程序（stdio）</option>
                      </select>
                    </label>
                    {editor.type === "http" ? (
                      <label>
                        服务地址
                        <input
                          aria-label="MCP 服务地址"
                          placeholder="https://example.com/mcp"
                          value={editor.url}
                          onChange={(e) =>
                            setEditor({ ...editor, url: e.target.value })
                          }
                        />
                      </label>
                    ) : (
                      <>
                        <label>
                          启动程序
                          <input
                            aria-label="MCP 启动程序"
                            placeholder="npx / uvx / Python 路径"
                            value={editor.command}
                            onChange={(e) =>
                              setEditor({ ...editor, command: e.target.value })
                            }
                          />
                        </label>
                        <label>
                          启动参数（JSON 数组）
                          <textarea
                            aria-label="MCP 启动参数"
                            value={editor.args}
                            onChange={(e) =>
                              setEditor({ ...editor, args: e.target.value })
                            }
                          />
                        </label>
                      </>
                    )}
                    <p className="mcp-footnote">
                      {editor.type === "http" ? "HTTP 请求头" : "环境变量"}
                      。已保存的值留空即可保留；删除一行可清除该项。
                    </p>
                    {editor.pairs.map((pair, index) => (
                      <div className="mcp-pair" key={index}>
                        <input
                          aria-label={`配置名称 ${index + 1}`}
                          placeholder={
                            editor.type === "http" ? "Authorization" : "API_KEY"
                          }
                          value={pair.key}
                          onChange={(e) =>
                            setEditor({
                              ...editor,
                              pairs: editor.pairs.map((p, i) =>
                                i === index ? { ...p, key: e.target.value } : p,
                              ),
                            })
                          }
                        />
                        <input
                          type="password"
                          autoComplete="off"
                          aria-label={`配置值 ${index + 1}`}
                          placeholder={
                            editor.server ? "已保存的值留空保留" : "值"
                          }
                          value={pair.value}
                          onChange={(e) =>
                            setEditor({
                              ...editor,
                              pairs: editor.pairs.map((p, i) =>
                                i === index
                                  ? { ...p, value: e.target.value }
                                  : p,
                              ),
                            })
                          }
                        />
                        <button
                          className="icon-button"
                          aria-label={`删除配置 ${index + 1}`}
                          onClick={() =>
                            setEditor({
                              ...editor,
                              pairs: editor.pairs.filter((_, i) => i !== index),
                            })
                          }
                        >
                          <X size={12} />
                        </button>
                      </div>
                    ))}
                    <button
                      className="mcp-add-pair"
                      onClick={() =>
                        setEditor({
                          ...editor,
                          pairs: [...editor.pairs, { key: "", value: "" }],
                        })
                      }
                    >
                      <Plus size={12} />
                      添加字段
                    </button>
                  </>
                )}
              </div>
              <p className="mcp-footnote">
                保存配置不会启动服务。测试连接会启动本地程序或访问远程地址；启用后，工具调用遵循会话权限。
              </p>
              <button
                className="primary mcp-save"
                disabled={locked || !editor.name.trim()}
                onClick={save}
              >
                {working === "save"
                  ? "保存中…"
                  : editor.entry
                    ? "安装并保存配置"
                    : "保存配置"}
              </button>
            </aside>
          )}
        </div>
        <footer className="mcp-footer">
          <span>内置 DDGS 多搜索源搜索可独立使用。</span>
          <span>当前项目 · {snapshot?.project.split(/[\\/]/).pop()}</span>
        </footer>
      </section>
    </div>
  );
}
