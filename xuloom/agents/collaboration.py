"""Delegation policy and counters over the existing native worker lifecycle."""
import uuid
from contextvars import ContextVar

DELEGATION_BUDGET = ContextVar("xuloom_delegation_budget", default=None)
MODES = ("solo", "auto", "ultra")
DEFAULTS = {
    "solo": {"concurrency": 0, "max_delegations": 0, "output_tokens": 0, "request_output_tokens": 0},
    "auto": {"concurrency": 3, "max_delegations": 12, "output_tokens": 1000000, "request_output_tokens": 64000},
    "ultra": {"concurrency": 8, "max_delegations": 32, "output_tokens": 1000000, "request_output_tokens": 64000},
}


def policy(mode="auto", settings=None):
    if mode not in MODES:
        raise ValueError("Collaboration mode must be solo, auto or ultra")
    settings = settings or {}
    values = dict(DEFAULTS[mode])
    if mode != "solo":
        # Retain the existing explicit host concurrency setting. Defaults are
        # initial product choices, not measurements of useful parallelism.
        if "agent_concurrency" in settings:
            values["concurrency"] = settings["agent_concurrency"]
            if type(values["concurrency"]) is int:
                values["max_delegations"] = max(values["max_delegations"], values["concurrency"] * 4)
        profiles = settings.get("collaboration_profiles", {})
        if not isinstance(profiles, dict):
            raise ValueError("Invalid collaboration profiles")
        overrides = profiles.get(mode, {})
        if not isinstance(overrides, dict) or overrides.keys() - values.keys():
            raise ValueError("Invalid collaboration profile limits")
        values.update(overrides)
        for name, maximum in (("concurrency", 80), ("max_delegations", 1000), ("output_tokens", 10000000), ("request_output_tokens", 10000000)):
            if type(values[name]) is not int or not 1 <= values[name] <= maximum:
                raise ValueError("Invalid collaboration limit: " + name)
    return {"mode": mode, **values}


def new_run(profile):
    return {"id": str(uuid.uuid4()), "submitted": 0, "budget": {"output_tokens": profile["output_tokens"]},
            "output_used": 0, "output_reserved": 0,
            "request_cap": profile["request_output_tokens"]}


def admit(context, count):
    profile = context.get("collaboration") or policy()
    require_delegation(context)
    run = context.setdefault("collaboration_run", new_run(profile))
    if run["submitted"] + count > profile["max_delegations"]:
        raise ValueError("本轮累计子任务已达协作预算上限；请汇总已有结果，不要继续派发。")
    if run["output_used"] + run["output_reserved"] >= run["budget"]["output_tokens"]:
        raise ValueError("本轮子代理输出预算已耗尽；请汇总已有结果。")
    # Atomic on the owning event loop; count admission attempts conservatively
    # even if registration/persistence subsequently fails.
    run["submitted"] += count


def require_delegation(context):
    if (context.get("collaboration") or policy())["mode"] == "solo":
        raise ValueError("Solo 独立模式禁止委派。请用户切换到 Auto 或 Ultra；完全访问不会解除此限制。")


def instructions(profile):
    mode = profile["mode"]
    principle = {
        "solo": "Complete work yourself. Do not spawn or resume subagents. If the user explicitly requests parallel workers, explain Solo is selected and ask them to switch the collaboration menu; do not switch it yourself.",
        "auto": "Delegate only when there is clearly independent work and its benefit exceeds coordination cost. Keep small, tightly coupled tasks together. Answer ordinary questions directly.",
        "ultra": "For complex work, actively check for independent investigations, alternative approaches and independent validation that can run in parallel. Continue your own useful work after dispatch. Answer simple tasks directly.",
    }[mode]
    return (f"# Collaboration: {mode}\n{principle}\n"
            "Never launch workers to fill a quota. Do not duplicate the same assignment unless the user explicitly asks for repeated independent trials. "
            "Give workers self-contained assignments instead of serially relaying every step. Wait for relevant results before final synthesis. "
            "There is no automatic queue: an over-capacity batch is rejected; split it and wait for capacity. "
            "Collaboration does not change tool permissions, reasoning effort or recursion limits. "
            f"Runtime limits per user turn: {profile['concurrency']} simultaneous workers, {profile['max_delegations']} admitted child executions, "
            f"{profile['output_tokens']} shared child output tokens, up to {profile['request_output_tokens']} per model response "
            "(not a currency or input-token budget). Shared means cumulative usage across all workers and responses, not a per-worker quota. "
            "A response is also capped by the model/API and remaining goal/shared budget; an upper bound is not a requested answer length.")
