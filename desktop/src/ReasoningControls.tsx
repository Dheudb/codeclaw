import type { ReasoningProfile } from "./types";

const labels: Record<string, string> = { default: "默认强度", none: "不推理", minimal: "最低", low: "低", medium: "中", high: "高", xhigh: "极高", max: "最高" };
const adapters = [
  ["openai", "OpenAI：reasoning_effort"], ["deepseek", "DeepSeek：thinking + effort"],
  ["qwen", "Qwen 兼容接口：enable_thinking"], ["qwen-local", "Qwen 本地：chat_template_kwargs"],
  ["anthropic", "Anthropic：自适应 Thinking + effort"], ["anthropic-budget", "Anthropic：固定 Thinking 预算"],
] as const;

export function ReasoningControls({ profile, disabled, change }: {
  profile?: ReasoningProfile; disabled: boolean;
  change: (value: Partial<Pick<ReasoningProfile, "thinking" | "effort">>) => void;
}) {
  if (!profile) return null;
  return <>
    {profile.adapter !== "openai" && <select className="thinking-picker" aria-label="Thinking 模式" title="独立于协作档位；默认不覆盖接口设置"
      disabled={disabled} value={profile.thinking} onChange={event => change({ thinking: event.target.value as ReasoningProfile["thinking"] })}>
      <option value="default">默认思考</option><option value="enabled">开启思考</option>
      {profile.canDisable && <option value="disabled">关闭思考</option>}
    </select>}
    {!!profile.efforts.length && <select className="reasoning-picker" aria-label="推理强度" title="仅显示当前模型声明支持的档位；关闭 Thinking 时不发送强度参数"
      disabled={disabled || profile.thinking === "disabled"} value={profile.effort} onChange={event => change({ effort: event.target.value })}>
      {["default", ...profile.efforts].map(value => <option key={value} value={value}>{labels[value] || value}</option>)}
    </select>}
  </>;
}

export function ReasoningSettings({ provider, profile, change }: {
  provider: string; profile?: ReasoningProfile; change: (profile?: ReasoningProfile) => void;
}) {
  function choose(adapter: ReasoningProfile["adapter"]) {
    change({ adapter, efforts: adapter === "openai" || adapter === "anthropic" ? ["low", "medium", "high"] : adapter === "deepseek" ? ["low", "high", "max"] : [],
      canDisable: !["openai", "anthropic"].includes(adapter), thinking: "default", effort: "default", budgetTokens: 2048 });
  }
  return <fieldset className="reasoning-settings"><legend>当前模型的思考能力</legend>
    <label>参数格式<select aria-label="推理参数格式" value={profile?.adapter || "unknown"} onChange={event => event.target.value === "unknown" ? change() : choose(event.target.value as ReasoningProfile["adapter"])}>
      <option value="unknown">未声明（不附加推理参数）</option>
      {adapters.filter(([id]) => provider === "anthropic" ? id.startsWith("anthropic") : !id.startsWith("anthropic")).map(([id, title]) => <option key={id} value={id}>{title}</option>)}
    </select></label>
    {profile && <>
      {profile.adapter !== "openai" && <label className="reasoning-toggle"><input type="checkbox" aria-label="模型支持关闭 Thinking" checked={profile.canDisable}
        onChange={event => change({ ...profile, canDisable: event.target.checked, thinking: !event.target.checked && profile.thinking === "disabled" ? "default" : profile.thinking })} />模型支持关闭 Thinking</label>}
      {profile.adapter !== "anthropic-budget" && <div role="group" aria-label="模型支持的推理强度" className="reasoning-efforts">
        <span>接口支持的强度</span>
        {[...(profile.adapter === "openai" ? ["none"] : []), "minimal", "low", "medium", "high", "xhigh", "max"].map(value => <label key={value}><input type="checkbox" checked={profile.efforts.includes(value)}
          onChange={event => { const efforts = event.target.checked ? [...profile.efforts, value] : profile.efforts.filter(item => item !== value); change({ ...profile, efforts, effort: efforts.includes(profile.effort) ? profile.effort : "default" }); }} />{labels[value]}</label>)}
      </div>}
      {profile.adapter === "anthropic-budget" && <label>Thinking token 预算<input type="number" aria-label="Thinking token 预算" min={1024} max={1000000} value={profile.budgetTokens}
        onChange={event => change({ ...profile, budgetTokens: Number(event.target.value) })} /></label>}
      <div className="reasoning-defaults"><ReasoningControls profile={profile} disabled={false} change={value => change({ ...profile, ...value })} /></div>
    </>}
    <p className="fine-print">按 API 连接与模型 ID 分别保存。请按供应商文档确认格式、开关和档位；兼容接口及本地服务可能不同。预设是编辑起点，不代表接口已验证支持。</p>
  </fieldset>;
}
