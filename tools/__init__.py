"""Deterministic calculator tools: pure, typed functions returning a ToolResult
(result, steps, citations). The LLM chooses a tool and its arguments; Python runs it."""

from tools.result import ToolResult

__all__ = ["ToolResult"]
