import { useState } from "react";
import { FileDiff, RefreshCw } from "lucide-react";
import type { FileChange, SessionView } from "./types";
import FileChanges from "./FileChanges";

export default function ChangesPanel({ session, invoke, onOpen }: {
  session: SessionView; invoke: (method: string) => Promise<any>; onOpen: (path: string, change?: FileChange) => void;
}) {
  const [git, setGit] = useState<{ status: string; diff: string }>();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  async function refresh() {
    setLoading(true); setError("");
    try { setGit(await invoke("changes")); } catch (reason) { setError(String(reason)); } finally { setLoading(false); }
  }
  return <>
    <div className="panel-title"><div><h3>文件变更</h3><p>自动记录会话修改，点击查看最新文件</p></div><FileDiff size={18} /></div>
    <div className="panel-scroll changes-scroll">
      {session.fileChanges?.length ? <FileChanges files={session.fileChanges} project={session.project} onOpen={onOpen} compact />
        : <div className="instruction-empty"><FileDiff size={26} /><p>写入或编辑文件后会自动显示在这里。</p></div>}
      <details className="git-changes"><summary>Git 工作区差异</summary>
        <button className="import-button" disabled={loading} onClick={refresh}><RefreshCw size={14} />{loading ? "读取中…" : "读取 Git 变更"}</button>
        {error && <p role="alert" className="error">{error}</p>}
        {git && <><pre className="git-status">{git.status || "没有 Git 变更，或当前目录不是 Git 仓库。"}</pre><pre className="diff-view">{git.diff}</pre></>}
      </details>
    </div>
  </>;
}
