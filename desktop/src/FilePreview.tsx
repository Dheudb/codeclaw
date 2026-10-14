import { useEffect, useState } from "react";
import { RefreshCw, X } from "lucide-react";
import type { FileChange } from "./types";
import Markdown from "./Markdown";
import { ChangeCounts, relativeFile } from "./FileChanges";

export default function FilePreview({ sessionId, project, path, change, version, close, onOpen }: {
  sessionId: string; project: string; path: string; change?: FileChange; version?: number;
  close: () => void; onOpen: (path: string) => void;
}) {
  const [file, setFile] = useState<{ content: string; path: string; size: number; truncated: boolean }>();
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [refresh, setRefresh] = useState(0);
  const markdown = /\.md(?:own)?$/i.test(path);
  const [tab, setTab] = useState(markdown ? "preview" : "source");
  useEffect(() => {
    let cancelled = false; setLoading(true); setError("");
    window.codeclaw.invoke("files.read", { session: sessionId, path }).then((value) => { if (!cancelled) setFile(value); })
      .catch((reason) => { if (!cancelled) { setFile(undefined); setError(String(reason)); } })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [sessionId, path, refresh, version]);
  useEffect(() => {
    const escape = (event: KeyboardEvent) => { if (event.key === "Escape") { event.stopPropagation(); close(); } };
    document.addEventListener("keydown", escape); return () => document.removeEventListener("keydown", escape);
  }, [close]);
  return <div className="overlay file-preview-overlay" onClick={close}>
    <section className="dialog file-preview" role="dialog" aria-modal="true" aria-label="文件预览" onClick={(event) => event.stopPropagation()}>
      <header><div><h2>{relativeFile(path, project)}</h2><small title={path}>{path}</small></div>
        <button className="icon-button" aria-label="刷新文件内容" disabled={loading} onClick={() => setRefresh(refresh + 1)}><RefreshCw size={16} /></button>
        <button className="icon-button" aria-label="关闭文件预览" autoFocus onClick={close}><X size={19} /></button></header>
      <nav className="file-preview-tabs" role="tablist" aria-label="文件视图">
        {[["preview", "预览"], ["source", "源码"], ["diff", "差异"]].filter(([id]) => id !== "preview" || markdown).map(([id, label]) =>
          <button key={id} role="tab" aria-selected={tab === id} onClick={() => setTab(id)}>{label}</button>)}
        {change && <ChangeCounts added={change.added} removed={change.removed} />}
        <small>文件内容读取自磁盘最新版本{file ? ` · ${file.size.toLocaleString()} 字节` : ""}</small>
      </nav>
      <div className="file-preview-body" role="tabpanel">
        {loading && <p role="status">正在读取文件…</p>}
        {error && <p role="alert" className="error">{error}</p>}
        {file && !loading && <>
          {file.truncated && <p className="preview-limitation">文件较大，仅展示前 500 KB。</p>}
          {tab === "preview" && <div className="markdown"><Markdown text={file.content} onOpenFile={onOpen} /></div>}
          {tab === "source" && <pre className="file-source">{file.content}</pre>}
          {tab === "diff" && (change?.statsAvailable ? <><p className="preview-limitation">这次工具修改时记录的差异。</p>
            {change.truncated && <p className="preview-limitation">差异较大，已截断显示。</p>}
            <pre className="file-diff">{change.diff ? change.diff.split("\n").map((line, index) => <div key={index} className={line.startsWith("+") ? "diff-added" : line.startsWith("-") ? "diff-removed" : line.startsWith("@@") ? "diff-hunk" : ""}>{line || " "}</div>) : "内容没有行级变化。"}</pre>
          </> : <p className="muted">这条历史没有保存修改前的内容，无法还原准确差异。可以查看最新文件内容。</p>)}
        </>}
      </div>
    </section>
  </div>;
}
