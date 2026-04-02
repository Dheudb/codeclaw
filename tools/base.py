from abc import ABC, abstractmethod
from typing import Type, Any, Dict
from pydantic import BaseModel

class BaseAgenticTool(ABC):
    """
    The fundamental interface that all CodeClaw tools must inherit. 
    It leverages Pydantic to tightly map abstract types to Anthropic's required tool JSON schema.
    """
    
    # Must be defined strictly by subclasses
    name: str = ""
    description: str = ""
    input_schema: Type[BaseModel] = BaseModel
    context: Dict[str, Any] = {}
    is_read_only: bool = False
    risk_level: str = "high"
    
    @classmethod
    def get_tool_definition(cls) -> Dict[str, Any]:
        """
        Dynamically generates the JSON schema specification for Anthropic API
        by interrogating the Pydantic input_schema Model.
        """
        if not cls.name or not cls.description:
            raise ValueError(f"Tool {cls.__name__} must define 'name' and 'description'")
            
        schema = cls.input_schema.model_json_schema()
            
        return {
            "name": cls.name,
            "description": cls.description,
            "input_schema": {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", [])
            }
        }
        
    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """
        Core async invocation method.
        Must be implemented by inherited tools.
        """
        pass

    def is_read_only_call(self, **kwargs) -> bool:
        return self.is_read_only

    def is_concurrency_safe_call(self, **kwargs) -> bool:
        return self.is_read_only_call(**kwargs)

    def get_risk_level(self, **kwargs) -> str:
        return self.risk_level

    def build_permission_summary(self, **kwargs) -> str:
        preview = []
        for key, value in list(kwargs.items())[:4]:
            rendered = str(value)
            if len(rendered) > 160:
                rendered = rendered[:157] + "..."
            preview.append(f"{key}: {rendered}")

        if not preview:
            return self.description

        return self.description + "\n" + "\n".join(preview)

    def prompt(self) -> str:
        return ""

    def validate_input(self, **kwargs):
        return None

    def check_permissions(self, permission_context: Dict[str, Any]):
        return None
        
    async def __call__(self, **kwargs):
        """
        Executes the tool with automated Pydantic schema coercion and validation.
        """
        sandbox = self.context.get("sandbox_manager")
        res_str = ""
        
        # Transparent Path Redirector Strategy: Protect LLM Context Limitations
        if sandbox and sandbox.active_sandbox:
            mapped_any = False
            for k in ["absolute_path", "TargetFile", "path", "cwd", "DirectoryPath"]:
                if k in kwargs and kwargs[k]:
                    if k == "cwd":
                        kwargs[k] = sandbox.resolve_working_directory(kwargs[k])
                    else:
                        kwargs[k] = sandbox.resolve_path(kwargs[k])
                    mapped_any = True
            
            # Special auto-targeting: generic commands missing explicit path get locked down
            if self.name in ["bash_tool", "grep_tool", "glob_tool"]:
                for k in ["path", "cwd"]:
                    if k not in kwargs or not kwargs[k]:
                        kwargs[k] = sandbox.resolve_working_directory(None)
                        mapped_any = True
                        
            if mapped_any:
                res_str = f"[✓ CodeClaw Sentinel: Action intercepted and safely routed to invisible sandbox sub-dimension '{sandbox.active_sandbox['branch']}']\\n"
        
        validated_input = self.input_schema(**kwargs)
        normalized_kwargs = validated_input.model_dump()
        validation_error = self.validate_input(**normalized_kwargs)
        if isinstance(validation_error, str) and validation_error.strip():
            raise ValueError(validation_error)

        raw_res = await self.execute(**normalized_kwargs)

        if isinstance(raw_res, dict) and {"ok", "content", "metadata", "is_error"}.issubset(raw_res.keys()) and res_str:
            enriched = dict(raw_res)
            enriched["content"] = res_str + "\n" + str(enriched.get("content", ""))
            metadata = dict(enriched.get("metadata", {}) or {})
            metadata["sandbox_branch"] = sandbox.active_sandbox["branch"] if sandbox and sandbox.active_sandbox else None
            metadata["sandbox_path"] = sandbox.active_sandbox["path"] if sandbox and sandbox.active_sandbox else None
            enriched["metadata"] = metadata
            return enriched
        
        if isinstance(raw_res, str) and res_str:
            return res_str + "\\n" + raw_res
        return raw_res
