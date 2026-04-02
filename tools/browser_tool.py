import os
from pydantic import BaseModel, Field
from codeclaw.tools.base import BaseAgenticTool
from codeclaw.core.tool_results import build_tool_result


class BrowserToolInput(BaseModel):
    action: str = Field(
        ...,
        description="One of: navigate, snapshot, click, type, wait, screenshot, close.",
    )
    url: str = Field(None, description="Target URL for navigate.")
    selector: str = Field(None, description="CSS selector for click or type.")
    text: str = Field(None, description="Input text for type.")
    seconds: float = Field(1.0, description="Seconds to wait for the wait action.")
    path: str = Field(None, description="Optional screenshot output path.")
    full_page: bool = Field(True, description="Whether screenshots should capture the full page.")
    clear: bool = Field(True, description="Whether type should replace existing input value.")


class BrowserTool(BaseAgenticTool):
    name = "browser_tool"
    description = "Controls a headless browser for webpage navigation, DOM inspection, clicking, typing, waiting, and screenshots. Use this when HTTP fetch is insufficient and real browser automation is required."
    input_schema = BrowserToolInput
    risk_level = "high"

    def prompt(self) -> str:
        return (
            "Use `browser_tool` when static fetch/search is insufficient and the task requires real page interaction, DOM inspection, waiting, or screenshots. "
            "Prefer `snapshot` before click/type so you have fresh page structure."
        )

    def validate_input(
        self,
        action: str,
        url: str = None,
        selector: str = None,
        text: str = None,
        seconds: float = 1.0,
        path: str = None,
        full_page: bool = True,
        clear: bool = True,
    ):
        valid_actions = {"navigate", "snapshot", "click", "type", "wait", "screenshot", "close"}
        if action not in valid_actions:
            return f"Unsupported browser_tool action '{action}'."
        if action == "navigate" and not url:
            return "url is required when action='navigate'."
        if action in {"click", "type"} and not selector:
            return f"selector is required when action='{action}'."
        if action == "type" and text is None:
            return "text is required when action='type'."
        return None

    def build_permission_summary(
        self,
        action: str,
        url: str = None,
        selector: str = None,
        text: str = None,
        seconds: float = 1.0,
        path: str = None,
        full_page: bool = True,
        clear: bool = True,
    ) -> str:
        text_preview = (text or "")[:120] + ("..." if text and len(text) > 120 else "")
        return (
            "Browser automation requested.\n"
            f"action: {action}\n"
            f"url: {url or '<none>'}\n"
            f"selector: {selector or '<none>'}\n"
            f"text_preview: {text_preview or '<empty>'}\n"
            f"seconds: {seconds}\n"
            f"path: {path or '<auto>'}"
        )

    async def execute(
        self,
        action: str,
        url: str = None,
        selector: str = None,
        text: str = None,
        seconds: float = 1.0,
        path: str = None,
        full_page: bool = True,
        clear: bool = True,
    ) -> dict:
        browser = self.context.get("browser_manager")
        if not browser:
            return build_tool_result(
                ok=False,
                content="Browser manager is unavailable.",
                metadata={"action": action},
                is_error=True,
            )

        try:
            if action == "navigate":
                if not url:
                    return build_tool_result(
                        ok=False,
                        content="url is required for navigate.",
                        metadata={"action": action},
                        is_error=True,
                    )
                result = await browser.navigate(url)
                return build_tool_result(
                    ok=True,
                    content=f"Navigated to {result['url']} ({result['title']})",
                    metadata={"action": action, **result},
                )

            if action == "snapshot":
                result = await browser.snapshot()
                interactive_lines = []
                for item in result.get("interactive_elements", []):
                    label = item.get("text") or "<no label>"
                    selector_hint = item.get("selector") or "<no id selector>"
                    interactive_lines.append(f"{item.get('index')}. <{item.get('tag')}> {label} [{selector_hint}]")
                content = (
                    f"Title: {result.get('title')}\n"
                    f"URL: {result.get('url')}\n"
                    f"Text preview:\n{result.get('text_preview', '')}\n\n"
                    f"Interactive elements:\n" + ("\n".join(interactive_lines) if interactive_lines else "None detected.")
                )
                return build_tool_result(
                    ok=True,
                    content=content,
                    metadata={
                        "action": action,
                        "url": result.get("url"),
                        "title": result.get("title"),
                        "interactive_count": len(result.get("interactive_elements", [])),
                    },
                )

            if action == "click":
                if not selector:
                    return build_tool_result(
                        ok=False,
                        content="selector is required for click.",
                        metadata={"action": action},
                        is_error=True,
                    )
                result = await browser.click(selector)
                return build_tool_result(
                    ok=True,
                    content=f"Clicked '{selector}' on {result['url']}.",
                    metadata={"action": action, **result},
                )

            if action == "type":
                if not selector:
                    return build_tool_result(
                        ok=False,
                        content="selector is required for type.",
                        metadata={"action": action},
                        is_error=True,
                    )
                if text is None:
                    return build_tool_result(
                        ok=False,
                        content="text is required for type.",
                        metadata={"action": action},
                        is_error=True,
                    )
                result = await browser.type(selector, text, clear=clear)
                return build_tool_result(
                    ok=True,
                    content=f"Typed into '{selector}' on {result['url']}.",
                    metadata={"action": action, "clear": clear, **result},
                )

            if action == "wait":
                result = await browser.wait(seconds)
                return build_tool_result(
                    ok=True,
                    content=f"Waited {seconds} second(s) on {result['url']}.",
                    metadata={"action": action, **result},
                )

            if action == "screenshot":
                normalized_path = os.path.abspath(path) if path else None
                result = await browser.screenshot(normalized_path, full_page=full_page)
                return build_tool_result(
                    ok=True,
                    content=f"Saved screenshot to {result['path']}.",
                    metadata={"action": action, "full_page": full_page, **result},
                )

            if action == "close":
                await browser.close()
                return build_tool_result(
                    ok=True,
                    content="Browser session closed.",
                    metadata={"action": action},
                )

            return build_tool_result(
                ok=False,
                content=f"Unsupported browser action '{action}'.",
                metadata={"action": action},
                is_error=True,
            )
        except Exception as e:
            return build_tool_result(
                ok=False,
                content=f"Browser action failed: {str(e)}",
                metadata={"action": action, "url": url or "", "selector": selector or ""},
                is_error=True,
            )
