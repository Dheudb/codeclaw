"""Shared reservation accounting for goal and delegated model output."""
from xuloom.features.goals import CURRENT_BUDGET, available_output
from xuloom.agents.collaboration import DELEGATION_BUDGET


def active_budgets(goal=None, delegation=None):
    result = []
    for budget in (goal or CURRENT_BUDGET.get(), delegation or DELEGATION_BUDGET.get()):
        if budget is not None and all(budget is not item for item in result):
            result.append(budget)
    return result


def reserve(maximum, budgets):
    amount = min([maximum] + [min(available_output(b), b.get("request_cap", maximum)) for b in budgets])
    if amount <= 0:
        raise RuntimeError("Model output-token budget is exhausted or reserved by active workers")
    for budget in budgets:
        budget["output_reserved"] += amount
    return amount


def settle(budgets, amount, reported):
    for budget in budgets:
        budget["output_used"] += reported if reported > 0 else amount
        budget["output_reserved"] -= amount
