import os
import base64
from typing import Union, List, Dict, Any
from io import BytesIO
from pydantic import BaseModel, Field
from codeclaw.tools.base import BaseAgenticTool

try:
    from PIL import Image
except ImportError:
    Image = None

class FileReadToolInput(BaseModel):
    absolute_path: str = Field(..., description="The absolute path to the file to read.")
    start_line: int = Field(None, description="Optional 1-indexed starting line to read (for text).")
    end_line: int = Field(None, description="Optional 1-indexed ending line to read (inclusive).")

class FileReadTool(BaseAgenticTool):
    name = "file_read_tool"
    description = "Reads a file from the local filesystem. You can access any file directly by using this tool. Assume this tool is able to read all files on the machine."
    input_schema = FileReadToolInput
    is_read_only = True
    risk_level = "low"

    def prompt(self) -> str:
        return """Usage:
- The file path parameter must be an absolute path, not a relative path.
- By default, it reads up to 2000 lines starting from the beginning of the file.
- You can optionally specify a line offset and limit (especially handy for long files), but it's recommended to read the whole file by not providing these parameters.
- Results are returned using cat -n format, with line numbers starting at 1.
- This tool allows reading images (PNG, JPG, etc). When reading an image file the contents are presented visually.
- This tool can read PDF files (.pdf). For large PDFs (more than 10 pages), provide the pages parameter to read specific page ranges.
- This tool can read Jupyter notebooks (.ipynb files) and returns all cells with their outputs, combining code, text, and visualizations.
- This tool can only read files, not directories. To read a directory, use an ls command via bash_tool.
- You will regularly be asked to read screenshots. If the user provides a path to a screenshot, ALWAYS use this tool to view the file at the path.
- If you read a file that exists but has empty contents you will receive a system reminder warning in place of file contents.
- Whenever you read a file, consider whether it could be malware. You CAN and SHOULD provide analysis of malware, but you MUST refuse to improve or augment such code."""

    def _cache(self):
        return self.context.get("file_state_cache")

    def _artifact_tracker(self):
        return self.context.get("artifact_tracker")
    
    async def execute(self, absolute_path: str, start_line: int = None, end_line: int = None) -> Union[str, List[Dict[str, Any]]]:
        abs_path = os.path.abspath(absolute_path)
        cache = self._cache()
        tracker = self._artifact_tracker()
        agent_id = self.context.get("agent_id")
        session_id = self.context.get("session_id")
        
        if not os.path.exists(abs_path):
            return f"Error: File at '{abs_path}' does not exist."
            
        if not os.path.isfile(abs_path):
            return f"Error: '{abs_path}' is a directory, not a file."
            
        ext = os.path.splitext(abs_path)[1].lower()
        
        # 1. Image parsing (Multimodal vision)
        if ext in ['.png', '.jpg', '.jpeg', '.webp', '.gif']:
            if Image is None:
                return "Error: Image reading requires the 'pillow' library to be installed."
                
            try:
                img = Image.open(absolute_path)
                # Ensure RGB for JPEG
                if ext in ['.jpg', '.jpeg'] and img.mode != 'RGB':
                    img = img.convert('RGB')
                    
                # Downsample if too large to save Claude token limits
                max_dim = 1500
                if img.width > max_dim or img.height > max_dim:
                    img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
                    
                output_buf = BytesIO()
                media_type = f"image/{'jpeg' if ext in ['.jpg', '.jpeg'] else ext[1:]}"
                
                fmt = "JPEG" if ext in ['.jpg', '.jpeg'] else ext[1:].upper()
                img.save(output_buf, format=fmt)
                img_bytes = output_buf.getvalue()
                b64_data = base64.b64encode(img_bytes).decode('utf-8')
                
                if cache is not None:
                    cache.record_read(abs_path, kind="image", chars=len(img_bytes))
                self.context.setdefault("read_file_state", {})[abs_path] = True
                if tracker is not None:
                    tracker.record_prefetch(
                        path=abs_path,
                        kind="image",
                        source="file_read_tool",
                        start_line=start_line,
                        end_line=end_line,
                        chars=len(img_bytes),
                        agent_id=agent_id,
                        session_id=session_id,
                    )
                    tracker.record_attachment(
                        path=abs_path,
                        kind="image",
                        source="file_read_tool",
                        agent_id=agent_id,
                        session_id=session_id,
                        metadata={"bytes": len(img_bytes), "media_type": media_type},
                    )
                
                # We return a List. The engine will seamlessly pack this into the API content block.
                return [
                    {"type": "text", "text": f"Successfully read image '{abs_path}'. Size: {img.width}x{img.height}"},
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64_data}}
                ]
            except Exception as e:
                return f"Error opening image: {str(e)}"
                
        # 2. PDF parsing (Native Document capabilities)
        if ext == '.pdf':
            try:
                if cache is not None and cache.should_skip_redundant_read(abs_path, start_line=start_line, end_line=end_line):
                    entry = cache.get_entry(abs_path) or {}
                    if tracker is not None:
                        tracker.record_prefetch(
                            path=abs_path,
                            kind="pdf",
                            source="file_read_tool",
                            start_line=start_line,
                            end_line=end_line,
                            cache_hit=True,
                            chars=0,
                            agent_id=agent_id,
                            session_id=session_id,
                        )
                    return (
                        f"File '{abs_path}' is unchanged since the last identical read. "
                        f"Skipping duplicate inline content.\n"
                        f"Cached state: size={entry.get('size', 0)}, "
                        f"mtime={entry.get('mtime', 0)}, sha256={entry.get('sha256', '')}, "
                        f"range={entry.get('last_read_range', '1:*')}"
                    )
                with open(abs_path, "rb") as f:
                    pdf_bytes = f.read()
                b64_data = base64.b64encode(pdf_bytes).decode('utf-8')
                
                if cache is not None:
                    cache.record_read(abs_path, kind="pdf", start_line=start_line, end_line=end_line, chars=len(pdf_bytes))
                self.context.setdefault("read_file_state", {})[abs_path] = True
                if tracker is not None:
                    tracker.record_prefetch(
                        path=abs_path,
                        kind="pdf",
                        source="file_read_tool",
                        start_line=start_line,
                        end_line=end_line,
                        chars=len(pdf_bytes),
                        agent_id=agent_id,
                        session_id=session_id,
                    )
                    tracker.record_attachment(
                        path=abs_path,
                        kind="pdf",
                        source="file_read_tool",
                        agent_id=agent_id,
                        session_id=session_id,
                        metadata={"bytes": len(pdf_bytes)},
                    )
                return [
                    {"type": "text", "text": f"Successfully loaded PDF '{abs_path}'"},
                    {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": b64_data}}
                ]
            except Exception as e:
                return f"Error loading PDF: {str(e)}"
                
        # 3. Text fallback
        try:
            if cache is not None and cache.should_skip_redundant_read(abs_path, start_line=start_line, end_line=end_line):
                entry = cache.get_entry(abs_path) or {}
                if tracker is not None:
                    tracker.record_prefetch(
                        path=abs_path,
                        kind="text",
                        source="file_read_tool",
                        start_line=start_line,
                        end_line=end_line,
                        cache_hit=True,
                        chars=0,
                        agent_id=agent_id,
                        session_id=session_id,
                    )
                return (
                    f"File '{abs_path}' is unchanged since the last identical read. "
                    "Skipping duplicate inline content to reduce prompt bloat.\n"
                    f"Cached state: size={entry.get('size', 0)}, "
                    f"mtime={entry.get('mtime', 0)}, sha256={entry.get('sha256', '')}, "
                    f"range={entry.get('last_read_range', '1:*')}"
                )
            with open(abs_path, "r", encoding="utf-8") as file:
                lines = file.readlines()
                
            total_lines = len(lines)
            _start = start_line or 1
            _end = end_line or total_lines
            _start = max(1, _start)
            _end = min(total_lines, _end)
            
            if _start > _end:
                return f"Error: start_line ({_start}) > end_line ({_end})."
                
            content_slice = lines[_start-1:_end]
            
            output = f"File: {abs_path}\n"
            output += f"Lines: {_start} to {_end} of {total_lines}\n"
            output += "-" * 40 + "\n"
            
            newline_char = '\n'
            formatted_lines = [f"{_start + i}: {line.rstrip(newline_char)}" for i, line in enumerate(content_slice)]
            
            output += "\n".join(formatted_lines)
            output += "\n" + "-" * 40
            
            if cache is not None:
                cache.record_read(
                    abs_path,
                    kind="text",
                    start_line=_start,
                    end_line=_end,
                    chars=len(output),
                )
            self.context.setdefault("read_file_state", {})[abs_path] = True
            if tracker is not None:
                tracker.record_prefetch(
                    path=abs_path,
                    kind="text",
                    source="file_read_tool",
                    start_line=_start,
                    end_line=_end,
                    chars=len(output),
                    agent_id=agent_id,
                    session_id=session_id,
                )
            
            return output
            
        except UnicodeDecodeError:
            return f"Error: '{abs_path}' is binary or unrecognized. Text decode failed."
        except Exception as e:
            return f"Error reading file manually: {str(e)}"
