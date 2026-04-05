import asyncio
import os
import shutil
import sys
import tempfile

from pydantic import BaseModel, Field

from codeclaw.context.tool_results import build_tool_result
from codeclaw.tools.base import BaseAgenticTool


class ReplToolInput(BaseModel):
    language: str = Field(..., description="Programming language to execute. Supported values: python, javascript.")
    code: str = Field(..., description="Code snippet to execute.")
    cwd: str = Field(None, description="Optional working directory for the REPL subprocess.")
    timeout_seconds: int = Field(20, description="Maximum runtime before the process is terminated.")


class ReplTool(BaseAgenticTool):
    name = "repl_tool"
    description = "Executes a short Python or JavaScript snippet in a subprocess with timeout protection. Useful for quick calculations, parsing, or small runtime experiments."
    input_schema = ReplToolInput
    risk_level = "high"

    def prompt(self) -> str:
        return (
            "Use `repl_tool` for small runtime experiments or computations when shell commands are awkward. "
            "Keep snippets short and avoid relying on long-lived process state."
        )

    def validate_input(
        self,
        language: str,
        code: str,
        cwd: str = None,
        timeout_seconds: int = 20,
    ):
        normalized = str(language or "").strip().lower()
        if normalized not in {"python", "javascript"}:
            return "language must be one of: python, javascript."
        if not str(code or "").strip():
            return "code is required."
        if int(timeout_seconds or 0) <= 0:
            return "timeout_seconds must be greater than 0."
        return None

    def build_permission_summary(
        self,
        language: str,
        code: str,
        cwd: str = None,
        timeout_seconds: int = 20,
    ) -> str:
        code_preview = code[:180] + ("..." if len(code) > 180 else "")
        return (
            "REPL execution requested.\n"
            f"language: {language}\n"
            f"cwd: {os.path.abspath(cwd) if cwd else os.getcwd()}\n"
            f"timeout_seconds: {timeout_seconds}\n"
            f"code_preview: {code_preview}"
        )

    async def execute(
        self,
        language: str,
        code: str,
        cwd: str = None,
        timeout_seconds: int = 20,
    ) -> dict:
        normalized = str(language or "").strip().lower()
        run_cwd = os.path.abspath(cwd) if cwd else os.getcwd()
        process = None
        if normalized == "python":
            interpreter = sys.executable
            suffix = ".py"
        else:
            interpreter = shutil.which("node")
            suffix = ".js"
            if not interpreter:
                return build_tool_result(
                    ok=False,
                    content="JavaScript execution requires `node` to be available on PATH.",
                    metadata={"language": normalized, "cwd": run_cwd},
                    is_error=True,
                )

        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=suffix, delete=False) as f:
                f.write(code)
                temp_path = f.name

            env = os.environ.copy()
            env["MPLBACKEND"] = "Agg"

            process = await asyncio.create_subprocess_exec(
                interpreter,
                temp_path,
                cwd=run_cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=float(timeout_seconds),
            )
            stdout_text = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
            stderr_text = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
            rendered = stdout_text.strip()
            if stderr_text.strip():
                rendered = (rendered + "\n--- STDERR ---\n" + stderr_text.strip()).strip()
            if not rendered:
                rendered = "REPL execution completed with no output."
            return build_tool_result(
                ok=process.returncode == 0,
                content=rendered,
                metadata={
                    "language": normalized,
                    "cwd": run_cwd,
                    "timeout_seconds": int(timeout_seconds),
                    "interpreter": interpreter,
                    "exit_code": process.returncode,
                },
                is_error=process.returncode != 0,
            )
        except asyncio.TimeoutError:
            try:
                process.kill()
            except Exception:
                pass
            return build_tool_result(
                ok=False,
                content=f"REPL execution timed out after {timeout_seconds} seconds.",
                metadata={
                    "language": normalized,
                    "cwd": run_cwd,
                    "timeout_seconds": int(timeout_seconds),
                    "interpreter": interpreter,
                },
                is_error=True,
            )
        except Exception as e:
            return build_tool_result(
                ok=False,
                content=f"Error during REPL execution: {str(e)}",
                metadata={
                    "language": normalized,
                    "cwd": run_cwd,
                    "timeout_seconds": int(timeout_seconds),
                    "interpreter": interpreter,
                },
                is_error=True,
            )
        finally:
            if temp_path:
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
