"""Tool registry for managing available tools."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
from pydantic import BaseModel
import json
import asyncio
import traceback


class ToolResult(BaseModel):
    """Result of a tool execution."""
    
    success: bool
    output: Any = None
    error: Optional[str] = None
    
    def to_message(self) -> str:
        """Convert result to a message string for the LLM."""
        if self.success:
            if isinstance(self.output, str):
                return self.output
            return json.dumps(self.output, indent=2, default=str)
        return f"Error: {self.error}"


@dataclass
class Tool:
    """Definition of a tool that can be called by the LLM."""
    
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable
    required_params: List[str] = field(default_factory=list)
    
    def to_openai_schema(self) -> Dict[str, Any]:
        """Convert tool to OpenAI-compatible function schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required_params,
                },
            },
        }
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        try:
            # Handle both sync and async functions
            if asyncio.iscoroutinefunction(self.function):
                result = await self.function(**kwargs)
            else:
                result = self.function(**kwargs)
            return ToolResult(success=True, output=result)
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            )


class ToolRegistry:
    """Registry for managing available tools."""
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
    
    def register_function(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        required: List[str] = None,
    ) -> Callable:
        """Decorator to register a function as a tool."""
        def decorator(func: Callable) -> Callable:
            tool = Tool(
                name=name,
                description=description,
                parameters=parameters,
                function=func,
                required_params=required or [],
            )
            self.register(tool)
            return func
        return decorator
    
    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def list_tools(self) -> List[Tool]:
        """List all registered tools."""
        return list(self._tools.values())
    
    def to_openai_schema(self) -> List[Dict[str, Any]]:
        """Get all tools in OpenAI-compatible schema."""
        return [tool.to_openai_schema() for tool in self._tools.values()]
    
    async def execute(self, name: str, **kwargs) -> ToolResult:
        """Execute a tool by name."""
        tool = self.get(name)
        if not tool:
            return ToolResult(success=False, error=f"Tool '{name}' not found")
        return await tool.execute(**kwargs)
