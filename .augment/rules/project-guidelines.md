---
type: "always_apply"
---

# Python Project Guidelines

## Architecture
- Backend: FastAPI / Django
- ORM: SQLAlchemy
- Database: PostgreSQL primary, Redis optional cache

## Dependencies
- Only approved libraries in requirements.txt
- No beta/unmaintained packages
- Always check license compliance

## Code Flow
- Business logic stays in services, not controllers
- Async DB/network calls
- Config/secrets separated from code
- Follow repository patterns for data access

## Deployment
- Dockerized deployment mandatory
- CI/CD with linting, tests, security scans
- Resource limits: 512MB–2GB per service

## Integrations
- Internal APIs: Auth, Logging, Payments
- External APIs: Stripe, Twilio, ML services
- Handle failures gracefully, retries where needed
