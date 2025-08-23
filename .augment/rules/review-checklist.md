---
type: "always_apply"
---

# Python Code Review Checklist

## Functionality
- Does the function do what it is supposed to?
- Are all edge cases handled? (None, empty, max/min)
- Network/db calls wrapped in try/except
- Async functions properly awaited

## Performance
- Avoid O(n^2) loops for large datasets
- Use generators/iterators for large data
- Cache repeated DB or API calls

## Security
- Input validation and sanitization done
- Secrets never hardcoded
- Proper auth/authz checks in place
- No sensitive info logged

## Testing
- Unit tests included
- Coverage ≥ 80%
- Edge cases explicitly tested
- Regression tests for critical modules

## Documentation
- Docstrings for all functions/classes
- Complex flows explained
- TODO/FIXME tracked in task manager

## Maintainability
- Functions < 50 lines if possible
- Cyclomatic complexity ≤ 10
- Reusable modules, no copy-paste
