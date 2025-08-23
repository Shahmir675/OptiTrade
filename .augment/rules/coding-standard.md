---
type: "always_apply"
---

# Python Coding Standards

## Language
- Python 3.13
- Always use type hints
- Prefer standard library over third-party when possible
- Async I/O for network/db calls
- Use uv always for package installs etc.

## Naming
- Variables/functions: snake_case
- Classes: PascalCase
- Constants: UPPER_CASE
- Modules/files: lowercase_with_underscores

## Formatting
- PEP8 enforced
- 4-space indentation
- Max line length: 100 chars
- Explicit imports only
- No trailing whitespace
- No comments

## Security & Best Practices
- Sanitize all user input
- Never store secrets in code; use environment variables
- Validate all external API responses
- Avoid mutable globals
- Use `with` context managers for files/connections

## Anti-Patterns
- Deeply nested loops/functions
- Catch-all exceptions without logging
- Re-inventing built-in functionality
- Hardcoding sensitive values
