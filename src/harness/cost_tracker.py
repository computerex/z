"""Cost tracking for LLM API usage."""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
import json


# LLM pricing (per 1M tokens) - adjust as needed
DEFAULT_PRICING = {
    # Z.AI GLM models
    "glm-4.7": {"input": 0.50, "output": 1.50},
    "glm-4": {"input": 0.50, "output": 1.50},
    "glm-4-plus": {"input": 1.00, "output": 3.00},
    "glm-4.6v": {"input": 0.50, "output": 1.50},  # Vision model
    # MiniMax models
    "MiniMax-M2.1": {"input": 0.14, "output": 0.56},
    "MiniMax-Text-01": {"input": 0.10, "output": 0.40},
    "abab6.5s-chat": {"input": 0.01, "output": 0.01},
    # Default fallback
    "default": {"input": 0.50, "output": 1.50},
}


@dataclass
class APICall:
    """Record of a single API call."""
    
    timestamp: datetime
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    duration_ms: float
    tool_calls: int = 0
    finish_reason: str = ""
    extra_usage: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "input_cost": self.input_cost,
            "output_cost": self.output_cost,
            "total_cost": self.total_cost,
            "duration_ms": self.duration_ms,
            "tool_calls": self.tool_calls,
            "finish_reason": self.finish_reason,
            "extra_usage": self.extra_usage,
        }


@dataclass
class CostSummary:
    """Summary of costs for a session or run."""
    
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_input_cost: float = 0.0
    total_output_cost: float = 0.0
    total_cost: float = 0.0
    total_duration_ms: float = 0.0
    total_tool_calls: int = 0
    extra_usage_totals: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_input_cost": round(self.total_input_cost, 6),
            "total_output_cost": round(self.total_output_cost, 6),
            "total_cost": round(self.total_cost, 6),
            "total_duration_ms": round(self.total_duration_ms, 2),
            "total_tool_calls": self.total_tool_calls,
            "extra_usage_totals": dict(self.extra_usage_totals),
            "avg_tokens_per_call": round(self.total_tokens / max(1, self.total_calls), 1),
            "avg_cost_per_call": round(self.total_cost / max(1, self.total_calls), 6),
        }
    
    def format_human(self) -> str:
        """Format for human display."""
        return (
            f"API Calls: {self.total_calls}\n"
            f"Tokens: {self.total_input_tokens:,} in / {self.total_output_tokens:,} out ({self.total_tokens:,} total)\n"
            f"Cost: ${self.total_input_cost:.4f} in / ${self.total_output_cost:.4f} out (${self.total_cost:.4f} total)\n"
            f"Tool Calls: {self.total_tool_calls}\n"
            f"Duration: {self.total_duration_ms / 1000:.2f}s"
        )


class CostTracker:
    """Track API costs and usage statistics."""
    
    def __init__(
        self,
        pricing: Optional[Dict[str, Dict[str, float]]] = None,
        on_update: Optional[Callable[[CostSummary], None]] = None,
        report_interval: int = 5,  # Report every N calls
    ):
        self.pricing = pricing or DEFAULT_PRICING
        self.on_update = on_update
        self.report_interval = report_interval
        
        self.calls: List[APICall] = []
        self.session_start = datetime.now()
        self._last_report_call_count = 0
    
    def get_pricing(self, model: str) -> Dict[str, float]:
        """Get pricing for a model. Supports exact, case-insensitive, and partial matching."""
        # Exact match
        if model in self.pricing:
            return self.pricing[model]
        
        # Case-insensitive match
        model_lower = model.lower()
        for key in self.pricing:
            if key.lower() == model_lower:
                return self.pricing[key]
        
        # Partial match (model name contains pricing key or vice versa)
        for key in self.pricing:
            if key.lower() in model_lower or model_lower in key.lower():
                return self.pricing[key]
        
        return self.pricing.get("default", {"input": 0.50, "output": 1.50})
    
    def record_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        duration_ms: float,
        tool_calls: int = 0,
        finish_reason: str = "",
        extra_usage: Optional[Dict[str, Any]] = None,
    ) -> APICall:
        """Record an API call."""
        pricing = self.get_pricing(model)
        
        # Cost per million tokens
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        
        call = APICall(
            timestamp=datetime.now(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=input_cost + output_cost,
            duration_ms=duration_ms,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            extra_usage=dict(extra_usage or {}),
        )
        
        self.calls.append(call)
        
        # Check if we should report
        if self.on_update and len(self.calls) - self._last_report_call_count >= self.report_interval:
            self._last_report_call_count = len(self.calls)
            self.on_update(self.get_summary())
        
        return call
    
    def get_summary(self) -> CostSummary:
        """Get summary of all tracked calls."""
        summary = CostSummary()
        
        for call in self.calls:
            summary.total_calls += 1
            summary.total_input_tokens += call.input_tokens
            summary.total_output_tokens += call.output_tokens
            summary.total_tokens += call.total_tokens
            summary.total_input_cost += call.input_cost
            summary.total_output_cost += call.output_cost
            summary.total_cost += call.total_cost
            summary.total_duration_ms += call.duration_ms
            summary.total_tool_calls += call.tool_calls
            for k, v in (call.extra_usage or {}).items():
                if k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    continue
                if isinstance(v, bool):
                    continue
                if isinstance(v, (int, float)):
                    summary.extra_usage_totals[k] = int(summary.extra_usage_totals.get(k, 0)) + int(v)

        return summary

    def get_cost_by_model(self) -> Dict[str, Dict[str, float]]:
        """Aggregate usage/cost by model for the current session."""
        out: Dict[str, Dict[str, float]] = {}
        for call in self.calls:
            row = out.setdefault(call.model, {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "input_cost": 0.0,
                "output_cost": 0.0,
                "total_cost": 0.0,
            })
            row["calls"] += 1
            row["input_tokens"] += call.input_tokens
            row["output_tokens"] += call.output_tokens
            row["total_tokens"] += call.total_tokens
            row["input_cost"] += call.input_cost
            row["output_cost"] += call.output_cost
            row["total_cost"] += call.total_cost
        return out
    
    def get_last_call(self) -> Optional[APICall]:
        """Get the most recent API call."""
        return self.calls[-1] if self.calls else None
    
    def reset(self) -> None:
        """Reset all tracking."""
        self.calls = []
        self.session_start = datetime.now()
        self._last_report_call_count = 0
    
    def to_dict(self) -> Dict:
        """Export all data as dictionary."""
        return {
            "session_start": self.session_start.isoformat(),
            "summary": self.get_summary().to_dict(),
            "calls": [call.to_dict() for call in self.calls],
        }
    
    def save(self, path: str) -> None:
        """Save tracking data to file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# Global tracker for session-wide cost tracking
_global_tracker: Optional[CostTracker] = None


def get_global_tracker() -> CostTracker:
    """Get or create global cost tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = CostTracker()
    return _global_tracker


def reset_global_tracker() -> None:
    """Reset the global tracker."""
    global _global_tracker
    _global_tracker = CostTracker()
