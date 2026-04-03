import os
import json
from pydantic import BaseModel, Field
from codeclaw.tools.base import BaseAgenticTool

class LspToolInput(BaseModel):
    action: str = Field(..., description="Action to perform: 'definition', 'references', or 'hover'")
    absolute_path: str = Field(..., description="Absolute path to the target file.")
    line: int = Field(..., description="0-indexed line number where the cursor should sit.")
    character: int = Field(..., description="0-indexed character offset from line start.")

class LspTool(BaseAgenticTool):
    name = "lsp_tool"
    description = "Semantic bridge to Language Servers for jumping to definitions, finding references, or getting type hover info. Uses 0-indexed positions."
    input_schema = LspToolInput
    is_read_only = True
    risk_level = "low"
    
    async def execute(self, action: str, absolute_path: str, line: int, character: int) -> str:
        lsp_manager = self.context.get("lsp_manager")
        if not lsp_manager:
            return "Error: Language Server Manager is currently offline or unconfigured. You must stick to 'grep_tool' blindly."
            
        abs_path = os.path.abspath(absolute_path)
        normalized_path = abs_path.replace('\\\\', '/')
        uri = f"file://{normalized_path}"
        
        if not os.path.exists(abs_path):
            return f"Error: Target file {abs_path} does not exist on disk."
            
        server_info = await lsp_manager.get_or_create_server(abs_path)
        if not server_info:
            return f"Error: Cannot find a supported Language Server backend for file '{abs_path}'."
            
        lsp, lang_id = server_info
            
        try:
            # Tell the server daemon we "opened" the file so its AST is parsed fresh
            text_content = ""
            with open(abs_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
                
            lsp.send_notification("textDocument/didOpen", {
                "textDocument": {
                    "uri": uri,
                    "languageId": "python", # Could be dynamically parsed from ext
                    "version": 1,
                    "text": text_content
                }
            })
            
            params = {
                "textDocument": {"uri": uri},
                "position": {"line": line, "character": character}
            }
            
            if action == 'references':
                params['context'] = {"includeDeclaration": True}
                res = await lsp.send_request("textDocument/references", params, timeout=15.0)
            elif action == 'definition':
                res = await lsp.send_request("textDocument/definition", params, timeout=10.0)
            elif action == 'hover':
                res = await lsp.send_request("textDocument/hover", params, timeout=5.0)
            else:
                return f"Error: Unrecognized LSP action '{action}'"
                
            if not res:
                return f"LSP daemon returned empty response (No references or definition found for line {line} char {character}). Is it a standard keyword?"
                
            # Stringify safely truncating massive arrays if any
            dump = json.dumps(res, indent=2)
            if len(dump) > 8000:
                dump = dump[:8000] + "\\n... [LSP TRUNCATED DUE TO SIZE]"
            return dump
            
        except Exception as e:
            return f"Error communicating via JSON-RPC LSP Bridge: {str(e)}"
