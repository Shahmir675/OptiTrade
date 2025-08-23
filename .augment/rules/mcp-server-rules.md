---
type: "always_apply"
---

# MCP Server Rules for Python Code Augmentation

## General Directive
- Always reference these rule files before generating or suggesting code:
    1. coding-standards.md
    2. project-guidelines.md
    3. review-checklist.md

- AI MUST produce code that fully conforms to these rules.
- Any suggestion violating these rules must be flagged and corrected automatically.

## Task Handling
- Break tasks into manageable steps using Sequential Thinking MCP server.
- For each step:
    1. Check relevant rules from coding-standards.md
    2. Ensure alignment with project-guidelines.md
    3. Validate against review-checklist.md

- Use Context7 MCP server to fetch the latest project docs or dependencies if needed.

## Validation
- Before returning any code:
    - Run logic check against coding standards.
    - Ensure maintainability and readability.
    - Confirm security and input validation measures.
    - Ensure async functions and DB calls follow project architecture.

## Edge Case Handling
- Explicitly handle None, empty, or malformed inputs.
- Validate external API and DB responses.
- Apply retries or fallback strategies for critical calls.

## Reporting
- Log any deviations from rules with reasoning.
- Suggest fixes automatically if rules are violated.
