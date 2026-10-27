import { useEffect, useState } from "react";
import { FileDiff, RefreshCw } from "lucide-react";
import type { FileChange, SessionView } from "./types";
import FileChanges from "./FileChanges";

export default function ChangesPanel({ session, invoke, onOpen }: {
  session: SessionView; invoke: (method: string) => Promise<any>; onOpen: (path: string, change?: FileChange) => void;
}) {
  const [git, setGit] = useState<{ available: boolean; files: FileChange[]; reason?: string; truncated?: boolean }>();
  const [loading, setLoading] = useState(false), [error, setError] = useState("");
  const [gitSession, setGitSession] = useState("");
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    if (gitSession !== session.id) { setGit(undefined); setError(""); setLoading(false); return; }
    let cancelled = false; setLoading(true); setError("");
    invoke("changes").then((value) => { if (!cancelled) setGit(value); }).catch((reason) => { if (!cancelled) setError(String(reason)); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [gitSession, session.id, session.project, session.fileChanges?.[0]?.updatedAt, revision]);
  return <>
    <div className="panel-title"><div><h3>文件变更</h3><p>点击查看修改前后差异，新增为绿，删除为红</p></div><FileDiff size={18} /></div>
    <div className="panel-scroll changes-scroll">
      {session.fileChanges?.length ? <FileChanges files={session.fileChanges} project={session.project} onOpen={onOpen} compact />
        : <div className="instruction-empty"><FileDiff size={26} /><p>写入或编辑文件后会自动显示在这里。</p></div>}
      <details key={session.id} className="git-changes" onToggle={(event) => { if (event.currentTarget.open) setGitSession(session.id); }}><summary>Git 工作区差异（可选）{git?.available ? ` · ${git.files.length} 个文件` : ""}</summary>
        <p className="muted">当前工作目录内相对 HEAD 的未提交变更，包含暂存、未暂存和新增文件。</p>
        {loading && !git && <p className="muted">正在检查 Git 工作区…</p>}
        {error && <p role="alert" className="error">{error}</p>}
        <button className="import-button" disabled={loading} onClick={() => setRevision(revision + 1)}><RefreshCw size={14} />{loading ? "读取中…" : "刷新 Git 变更"}</button>
        {git?.available && (git.files.length ? <FileChanges files={git.files} project={session.project} onOpen={onOpen} compact /> : <p className="muted">工作区没有未提交变更。</p>)}
        {git?.truncated && <p className="muted">文件较多，仅显示前 500 项。</p>}
        {git && !git.available && <p className="muted git-unavailable">{git.reason}。会话文件变更仍会正常记录。</p>}
      </details>
    </div>
  </>;
}
