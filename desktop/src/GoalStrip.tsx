import { useEffect, useState } from "react";
import { Target, PauseCircle, PlayCircle, Trash2, Maximize2, LoaderCircle } from "lucide-react";
import type { SessionView } from "./types";

const labels: Record<string, string> = { active: "进行中的目标", paused: "已暂停的目标", blocked: "目标需要处理", interrupted: "已中断的目标", awaiting_review: "目标等待验收" };
function duration(seconds: number) {
  const value = Math.max(0, Math.floor(seconds)), hours = Math.floor(value / 3600), minutes = Math.floor(value % 3600 / 60);
  return `${hours ? hours + "小时" : ""}${minutes ? minutes + "分" : ""}${value % 60}秒`;
}

export default function GoalStrip({ goal, busy, invoke, onError, details }: {
  goal: NonNullable<SessionView["goal"]>; busy: boolean;
  invoke: (method: string, params?: Record<string, unknown>) => Promise<any>;
  onError: (error: unknown) => void; details: () => void;
}) {
  const [now, setNow] = useState(Date.now()), [pending, setPending] = useState(false);
  useEffect(() => {
    setNow(Date.now());
    if (!busy || goal.status !== "active") return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [busy, goal.status, goal.round_started_at]);
  if (["completed", "cancelled", "budget_exhausted"].includes(goal.status)) return null;
  const elapsed = (goal.elapsed_seconds || 0) + (busy && goal.status === "active" && goal.phase === "executing" && goal.round_started_at ? Math.max(0, now / 1000 - goal.round_started_at) : 0);
  async function control(action: "pause" | "cancel" | "resume") {
    setPending(true);
    try { await invoke(action === "resume" ? "goal.run" : "goal.control", action === "resume" ? {} : { action }); }
    catch (error) { onError(error); }
    finally { setPending(false); }
  }
  return <div className="goal-strip" role="region" aria-label="持续目标">
    <Target size={15} />
    <button className="goal-strip-summary" aria-label="管理当前目标" title={goal.objective} onClick={details}><strong>{labels[goal.status] || "当前目标"}</strong><span>{goal.objective}</span></button>
    <span className="goal-strip-time" title="累计执行时间；暂停时间不计入">{duration(elapsed)}</span>
    {pending && <LoaderCircle size={14} className="spin" />}
    <button className="icon-button" aria-label="取消持续目标" title="取消持续目标" disabled={pending} onClick={() => void control("cancel")}><Trash2 size={14} /></button>
    {busy && goal.status === "active" ? <button className="icon-button" aria-label="暂停持续目标" title="暂停持续目标" disabled={pending} onClick={() => void control("pause")}><PauseCircle size={15} /></button>
      : goal.status !== "awaiting_review" && <button className="icon-button" aria-label="恢复持续目标" title="恢复持续目标" disabled={pending || busy} onClick={() => void control("resume")}><PlayCircle size={15} /></button>}
    <button className="icon-button" aria-label="查看目标详情" title="详情、验收与预算" onClick={details}><Maximize2 size={14} /></button>
  </div>;
}
