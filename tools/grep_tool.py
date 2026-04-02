import asyncio
import os
from pydantic import BaseModel, Field
from codeclaw.tools.base import BaseAgenticTool

class GrepToolInput(BaseModel):
    query: str = Field(..., description="The regex pattern or string to search for.")
    path: str = Field(None, description="The directory to search in. Defaults to current directory.")
    include: str = Field(None, description="Optional glob file pattern to narrow search (e.g. *.py)")
    head_limit: int = Field(250, description="Max lines of matches to return initially to save API token bounds.")
    offset: int = Field(0, description="Skip offset. Use this if previous search was paginated.")

class GrepTool(BaseAgenticTool):
    name = "grep_tool"
    description = "A powerful search tool built on ripgrep. ALWAYS use grep_tool for search tasks. NEVER invoke grep or rg as a bash_tool command."
    input_schema = GrepToolInput
    is_read_only = True
    risk_level = "low"

    def prompt(self) -> str:
        return """Usage:
- Supports full regex syntax (e.g., "log.*Error", "function\\s+\\w+")
- Filter files with include parameter using glob patterns (e.g., "*.js", "**/*.tsx")
- Use agent_tool for open-ended searches requiring multiple rounds
- Pattern syntax: Uses ripgrep (not grep) — literal braces need escaping (use interface\\{\\} to find interface{} in Go code)
- Multiline matching: By default patterns match within single lines only. For cross-line patterns like struct \\{[\\s\\S]*?field, use multiline patterns.
- Results are capped for responsiveness; when truncation occurs, refine your search pattern."""
    
    async def execute(self, query: str, path: str = None, include: str = None, head_limit: int = 250, offset: int = 0) -> str:
        search_path = path or os.getcwd()
        search_path = os.path.abspath(search_path)
        
        # Build the command pushing max defenses (-n forces ripgrep to print file:line)
        cmd = ["rg", "-n", "--max-columns", "500", query, search_path]
        if include:
            cmd.extend(["-g", include])
            
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=40)
            
            # ripgrep returns 0 for matches, 1 for no matches, 2 for errors
            if process.returncode == 0:
                output = stdout.decode('utf-8', errors='replace')
                lines = output.splitlines()
                
                # We extract all file names hit
                mtimes = {}
                for line in lines:
                    if ":" in line:
                        f_name = line.split(":", 1)[0]
                        if f_name not in mtimes:
                            try:
                                mtimes[f_name] = os.path.getmtime(f_name)
                            except Exception:
                                mtimes[f_name] = 0
                                
                # Sort exactly like Claude Code:
                # Group logic - group by mtime descending (newest touched at top), then by original find order internally
                # to keep multi-line matches linked.
                indexed_lines = list(enumerate(lines))
                
                def sort_key(item):
                    idx, line = item
                    if ":" not in line:
                        return (0, line, idx)
                    f_name = line.split(":", 1)[0]
                    # Python sorting is ascending, so we do negative mtime to reverse sort
                    return (-mtimes.get(f_name, 0), f_name, idx)
                    
                indexed_lines.sort(key=sort_key)
                
                # strip indices back out
                sorted_lines = [item[1] for item in indexed_lines]
                
                # Enforce Offset (skip mechanism)
                if offset > 0:
                    sorted_lines = sorted_lines[offset:]
                    
                # Enforce Truncation bounds
                if len(sorted_lines) > head_limit:
                    visible_block = sorted_lines[:head_limit]
                    truncated = "\\n".join(visible_block)
                    next_offset = offset + head_limit
                    return f"{truncated}\\n\\n... [Pagination applied. Showing {head_limit} items. Next offset: {next_offset}. Supply offset={next_offset} to view more matches.]"
                
                if offset > 0 and len(sorted_lines) == 0:
                     return "No more pagination matches at this internal offset value."
                
                return "\\n".join(sorted_lines)
                
            elif process.returncode == 1:
                return "No matches found."
            else:
                return f"Ripgrep search internal logic crash. stderr: {stderr.decode('utf-8', errors='replace')}"
                
        except asyncio.TimeoutError:
            process.kill()
            return "Error: Grep search timed out after 40 seconds. Your query may have been too generic or scanning a massive binary blob."
        except FileNotFoundError:
            return "Warning: 'ripgrep' (`rg`) is missing! Required for fast codebase indexing. Please have the human administrator globally install `ripgrep`."
        except Exception as e:
            return f"Search execution crashed: {str(e)}"
