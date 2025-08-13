# Security Policy

We take the security of HASN-AI seriously. This document outlines how to report vulnerabilities and our expectations for responsible disclosure.

## Supported versions

We aim to support the default branch (main) and the most recent tagged release. Older versions may not receive security updates.

## Reporting a vulnerability

- Email: security reports can be sent to the maintainers privately. If an email is not available publicly, open a private advisory (GitHub Security Advisories) or create a minimal non-sensitive issue requesting a secure contact channel.
- Please include:
  - Affected version/commit hash
  - Reproduction steps or proof-of-concept
  - Impact assessment
  - Any suggested mitigations
- Do not create public issues containing exploit details.

## Responsible disclosure timeline

- We will acknowledge your report within 7 days.
- We aim to provide an initial assessment within 14 days.
- A fix or mitigation timeline will be coordinated based on severity and impact.

## Security best practices for contributors

- Run security scans locally before submitting PRs:
  - Filesystem scan: `make trivy-fs`
  - Container image scan: `make trivy-image`
- Avoid committing secrets. Consider using tools like `git-secrets` or `pre-commit` hooks.
- Keep dependencies up to date; pin versions where appropriate.

## Dependency and container security

- Dependencies are pinned in `requirements.txt`.
- Docker images should be built with `make docker-build` and scanned with Trivy.

## Contact

If you believe you have found a vulnerability and cannot use the channels above, you can create a GitHub issue requesting a secure contact path (avoid sensitive details).
