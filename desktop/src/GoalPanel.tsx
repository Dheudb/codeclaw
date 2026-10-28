import { useEffect, useState } from "react";

const labels: Record<string, string> = { active: "执行中", paused: "已暂停", blocked: "受阻", interrupted: "已中断", awaiting_review: "等待验收", completed: "已完成", cancelled: "已取消", budget_exhausted: "预算耗尽" };
export default function GoalPanel({ invoke, busy, onError, compact = false, initialObjective = "", skills, onCreated, onStarted }: {
  invoke: (method: string, params?: Record<string, unknown>) => Promise<any>; busy: boolean; onError: (error: unknown) => void;
  compact?: boolean; initialObjective?: string; skills?: string[]; onCreated?: () => void; onStarted?: () => void;
}) {
  const [goal, setGoal] = useState<any>(null), [jobs, setJobs] = useState<any[]>([]);
  const [objective, setObjective] = useState(initialObjective), [criteria, setCriteria] = useState("");
  const [rounds, setRounds] = useState(20), [minutes, setMinutes] = useState(30), [tokens, setTokens] = useState(50000);
  const [message, setMessage] = useState(""), [delay, setDelay] = useState(30), [daily, setDaily] = useState(false), [pending, setPending] = useState(false);
  const refresh = async () => { const [g, j] = await Promise.all([invoke("goal.get"), invoke("schedule.list")]); setGoal(g); setJobs(j); };
  useEffect(() => { let live = true; const load = () => refresh().catch((error) => { if (live) onError(error); }); load(); const timer = setInterval(load, 3000); return () => { live = false; clearInterval(timer); }; }, []);
  async function action(method: string, params: Record<string, unknown> = {}) { setPending(true); try { await invoke(method, params); await refresh(); } catch (e) { onError(e); } finally { setPending(false); } }
  const terminal = goal && ["completed", "cancelled", "budget_exhausted"].includes(goal.status);
  return <div className="panel-scroll goal-panel">
    <div className="panel-title"><div><h3>{compact ? "目标模式" : "目标与唤醒"}</h3><p>沿用此会话的模型、Skills、权限和执行记录。</p></div></div>
    {goal && <section className="goal-card"><strong>{goal.objective}</strong><p>{labels[goal.status] || goal.status} · {goal.rounds}/{goal.budget.rounds} 轮 · 用时 {Math.ceil(goal.elapsed_seconds)} 秒</p><p>输出预算 {goal.output_used}/{goal.budget.output_tokens}</p><p>{goal.status === "awaiting_review" ? "模型已提交结果，请检查验收条件后确认。" : goal.reason}</p><ul>{goal.criteria.map((c: string, i: number) => <li key={i}>{c}</li>)}</ul>
      {!terminal && <div className="goal-actions">
        <button disabled={pending || busy} onClick={() => action("goal.run")}>恢复目标</button>
        <button disabled={pending || !busy} onClick={() => action("goal.control", { action: "pause" })}>暂停</button>
        <button disabled={pending} onClick={() => action("goal.control", { action: "cancel" })}>取消目标</button>
        {goal.status === "awaiting_review" && <button disabled={pending || busy} onClick={() => action("goal.control", { action: "accept", reason: "用户在目标面板确认验收通过" })}>确认验收通过</button>}
      </div>}
    </section>}
    {(!goal || terminal) && <form className="goal-card" onSubmit={async (e) => { e.preventDefault(); setPending(true); try { await invoke("goal.create", { objective, criteria: criteria.split("\n").filter(Boolean), rounds, seconds: minutes * 60, output_tokens: tokens, skills }); onCreated?.(); await invoke("goal.run"); await refresh(); onStarted?.(); } catch (error) { onError(error); } finally { setPending(false); } }}>
      <h4>新目标</h4><label>目标<textarea aria-label="目标" required value={objective} onChange={e => setObjective(e.target.value)} /></label>
      <label>验收条件（每行一项）<textarea aria-label="验收条件（每行一项）" required value={criteria} onChange={e => setCriteria(e.target.value)} /></label>
      <label>最多轮数<input aria-label="最多轮数" type="number" min="1" max="10000" value={rounds} onChange={e => setRounds(Number(e.target.value))} /></label>
      <label>执行预算（分钟）<input aria-label="执行预算（分钟）" type="number" min="1" max="10080" value={minutes} onChange={e => setMinutes(Number(e.target.value))} /></label>
      <label>输出 Token 预算<input aria-label="输出 Token 预算" type="number" min="1" max="10000000" value={tokens} onChange={e => setTokens(Number(e.target.value))} /></label>
      <small>文字验收条件由你最终确认；模型声称完成不会自动视为通过。暂停时间不计入执行预算。</small>
      <button disabled={busy || pending} type="submit">创建并开始</button>
    </form>}
    {!compact && <><form className="goal-card" onSubmit={e => { e.preventDefault(); action("schedule.add", { prompt: message, due_at: Date.now() / 1000 + delay * 60, ...(daily ? { interval: 86400 } : {}) }); }}>
      <h4>稍后向此会话发送消息</h4><textarea required aria-label="定时消息" value={message} onChange={e => setMessage(e.target.value)} />
      <label>多少分钟后<input type="number" min="0" required value={delay} onChange={e => setDelay(Number(e.target.value))} /></label>
      <label><input type="checkbox" checked={daily} onChange={e => setDaily(e.target.checked)} />之后每 24 小时重复</label>
      <small>应用或 CLI 唤醒服务运行时投递；关机期间不执行，重启后合并错过的周期。忙碌时排队，仍遵守会话审批。</small>
      <button type="submit" disabled={pending}>保存唤醒</button>
    </form>
    <h4>定时消息</h4>{jobs.map(job => <section key={job.id} className="goal-card"><strong>{job.prompt}</strong><p>{labels[job.status] || job.status} · {new Date(job.due_at * 1000).toLocaleString()}</p><small>{job.cwd}</small><p>{job.reason}</p>{!["completed", "cancelled"].includes(job.status) && <div className="goal-actions"><button disabled={pending} onClick={() => action("schedule.control", { id: job.id, action: job.status === "paused" ? "resume" : "pause" })}>{job.status === "paused" ? "恢复唤醒" : "暂停唤醒"}</button><button disabled={pending} onClick={() => action("schedule.control", { id: job.id, action: "cancel" })}>取消唤醒</button></div>}</section>)}</>}
  </div>;
}
