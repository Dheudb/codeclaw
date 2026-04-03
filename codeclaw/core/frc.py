"""
Function Result Clearing (FRC)

Precisely clears old tool_result content from the message history,
keeping only the most recent N results intact. This prevents context
window bloat without destroying conversation structure.

Unlike the broad `micro_compact_tool_results` in MemoryCompactor, FRC:
- Preserves tool_use / tool_result pairing (required by Anthropic API)
- Replaces content with a brief summary, not a truncation
- Tracks which results have been cleared to avoid re-processing
- Works incrementally each turn rather than in bulk
"""
import copy
import json
from typing import Optional


CLEARED_MARKER = "[FRC: tool result cleared — see earlier context or re-run tool]"


def _extract_tool_result_summary(content) -> str:
    """Extract a one-line summary from tool result content."""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "image":
                    parts.append("[image]")
            elif isinstance(block, str):
                parts.append(block)
        text = " ".join(parts)
    else:
        text = str(content)
    text = " ".join(text.split())
    if len(text) <= 120:
        return text
    return text[:100] + "..."


def _content_size(content) -> int:
    """Estimate the character size of tool result content."""
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for block in content:
            if isinstance(block, dict):
                total += len(str(block.get("text", ""))) + len(str(block.get("content", "")))
            else:
                total += len(str(block))
        return total
    return len(str(content))


def clear_old_function_results(
    messages: list,
    *,
    preserve_recent_results: int = 6,
    min_content_chars: int = 200,
    cleared_ids: Optional[set] = None,
) -> tuple[list, int, set]:
    """
    Clear old tool_result content from messages, preserving recent ones.

    Returns:
        (new_messages, num_cleared, updated_cleared_ids)
    """
    if cleared_ids is None:
        cleared_ids = set()

    all_tool_result_positions = []
    for msg_idx, msg in enumerate(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block_idx, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_result":
                continue
            tool_use_id = block.get("tool_use_id", "")
            if tool_use_id in cleared_ids:
                continue
            if _content_size(block.get("content", "")) < min_content_chars:
                continue
            all_tool_result_positions.append((msg_idx, block_idx, tool_use_id))

    if len(all_tool_result_positions) <= preserve_recent_results:
        return messages, 0, cleared_ids

    to_clear = all_tool_result_positions[:-preserve_recent_results]
    if not to_clear:
        return messages, 0, cleared_ids

    cloned = copy.deepcopy(messages)
    new_cleared = set()
    num_cleared = 0

    for msg_idx, block_idx, tool_use_id in to_clear:
        block = cloned[msg_idx]["content"][block_idx]
        original_content = block.get("content", "")
        summary = _extract_tool_result_summary(original_content)
        char_count = _content_size(original_content)
        block["content"] = (
            f"{CLEARED_MARKER}\n"
            f"Original size: {char_count} chars\n"
            f"Summary: {summary}"
        )
        new_cleared.add(tool_use_id)
        num_cleared += 1

    return cloned, num_cleared, cleared_ids | new_cleared


def estimate_frc_savings(messages: list, *, preserve_recent_results: int = 6) -> dict:
    """Estimate how much could be saved by running FRC."""
    all_results = []
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                all_results.append(_content_size(block.get("content", "")))

    if len(all_results) <= preserve_recent_results:
        return {"clearable_results": 0, "estimated_chars_saved": 0}

    clearable = all_results[:-preserve_recent_results]
    return {
        "clearable_results": len(clearable),
        "estimated_chars_saved": sum(clearable),
        "total_results": len(all_results),
    }
