import os
import time
import uuid
from typing import Any, Dict


def build_tool_result(
    *,
    ok: bool,
    content: str,
    metadata: Dict[str, Any] = None,
    is_error: bool = False,
) -> Dict[str, Any]:
    return {
        "ok": ok,
        "content": content,
        "metadata": metadata or {},
        "is_error": is_error,
    }


def serialize_tool_result(
    payload: Any,
    *,
    spill_threshold_chars: int = 5000,
    artifact_dir: str = ".codeclaw/tool-results",
    spill_recorder=None,
) -> str:
    if isinstance(payload, dict) and {"ok", "content", "metadata", "is_error"}.issubset(payload.keys()):
        content = str(payload.get("content", ""))
        metadata = dict(payload.get("metadata", {}) or {})
        is_error = bool(payload.get("is_error", False))
        ok = bool(payload.get("ok", False))

        if content and len(content) > spill_threshold_chars:
            try:
                os.makedirs(artifact_dir, exist_ok=True)
                file_path = os.path.abspath(
                    os.path.join(
                        artifact_dir,
                        f"tool-result-{int(time.time())}-{uuid.uuid4().hex[:8]}.txt",
                    )
                )
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                metadata["content_spilled"] = True
                metadata["content_spilled_path"] = file_path
                metadata["original_content_chars"] = len(content)
                if callable(spill_recorder):
                    try:
                        spill_recorder(
                            file_path=file_path,
                            original_content_chars=len(content),
                            metadata=metadata,
                        )
                    except Exception:
                        pass
                preview = content[:800]
                content = (
                    "Tool result was too large for inline context and has been written to disk.\n"
                    f"Reference path: {file_path}\n"
                    f"Preview:\n{preview}"
                )
            except Exception as e:
                metadata["content_spill_error"] = str(e)

        lines = [f"Status: {'ok' if ok else 'error'}"]
        if metadata:
            lines.append("Metadata:")
            for key, value in metadata.items():
                lines.append(f"- {key}: {value}")
        if content:
            lines.append("Content:")
            lines.append(content)

        rendered = "\n".join(lines).strip()
        if is_error and not rendered.startswith("Status: error"):
            rendered = "Status: error\n" + rendered
        return rendered or ("Status: ok" if ok else "Status: error")

    return str(payload)
