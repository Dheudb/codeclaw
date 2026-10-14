import { useState } from "react";
import { ChevronDown, FileDiff } from "lucide-react";
import type { FileChange } from "./types";

export function relativeFile(path: string, project: string) {
  const value = path.replaceAll("\\", "/"), root = project.replaceAll("\\", "/").replace(/\/$/, "") + "/";
  const windows = /^[a-z]:\//i.test(value);
  return (windows ? value.toLowerCase().startsWith(root.toLowerCase()) : value.startsWith(root)) ? value.slice(root.length) : path;
}
export function ChangeCounts({ added, removed }: { added: number | null; removed: number | null }) {
  return added === null || removed === null ? <small className="file-counts-unknown">未记录行数</small>
    : <span className="file-counts"><span className="lines-added">+{added}</span><span className="lines-removed">-{removed}</span></span>;
}
export default function FileChanges({ files, project, onOpen, compact = false }: {
  files: FileChange[]; project: string; onOpen: (path: string, change?: FileChange) => void; compact?: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  if (!files.length) return null;
  const known = files.filter((file) => file.statsAvailable);
  return <section className={"file-changes " + (compact ? "compact" : "")} aria-label="文件变更">
    <header><FileDiff size={17} /><div><strong>{compact ? "会话修改的文件" : `已修改 ${files.length} 个文件`}</strong>
      <ChangeCounts added={known.length ? known.reduce((sum, file) => sum + (file.added || 0), 0) : null} removed={known.length ? known.reduce((sum, file) => sum + (file.removed || 0), 0) : null} />
      {known.length !== files.length && <small>部分历史未记录行数</small>}</div></header>
    {(expanded || compact ? files : files.slice(0, 3)).map((file) => <button key={file.path} className="changed-file-row" title={file.path}
      aria-label={"打开文件 " + relativeFile(file.path, project)} onClick={() => onOpen(file.path, file)}>
      <span>{relativeFile(file.path, project)}</span><ChangeCounts added={file.added} removed={file.removed} /></button>)}
    {!compact && files.length > 3 && <button className="show-more-files" onClick={() => setExpanded(!expanded)}>
      {expanded ? "收起文件列表" : `再显示 ${files.length - 3} 个文件`}<ChevronDown size={13} /></button>}
  </section>;
}
