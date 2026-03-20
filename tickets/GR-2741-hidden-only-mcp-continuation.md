# GR-2741: Prevent Premature Turn End on Hidden-Only Model Output During MCP Flows

## Problem
When using `glm-4.7` with MCP-heavy workflows (e.g., Playwright), the model can return a response that contains only hidden reasoning and no visible content/tool XML.  
The harness may treat that as a completed turn and return to prompt, which looks like the agent "stopped on its own."

## Scope
- Detect hidden-only assistant outputs more reliably.
- Retry with a system nudge to force either:
  - one valid tool call, or
  - a concise visible completion.
- Keep behavior bounded (small retry cap).

## Acceptance Criteria
1. If provider response has empty visible content and reasoning exists, harness retries instead of ending turn immediately.
2. Retry counter resets after a successful parsed tool call.
3. Retry does not loop indefinitely.
4. Existing normal completion behavior remains unchanged when visible content is present.

## Implementation Notes
- File: `src/harness/cline_agent.py`
- Use raw provider content (`response.content`) as the primary hidden-only signal to avoid false negatives from post-processed buffers.
