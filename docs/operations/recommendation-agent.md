# Recommendation Agent deployment boundary

Deploy the Recommendation Agent by immutable image digest on private ingress;
there is no public route. Backend authenticates with the injected Bearer
service credential. Egress is restricted to the configured private HTTPS
inference origin and DNS; redirects, environment proxies, arbitrary request
hosts, databases, web/LLM providers, and Docker socket access are disallowed.

Run as UID/GID `10001:10001` with a read-only root filesystem, one worker, the
documented request/concurrency/deadline limits, and no model, database,
training, or host volume. If a platform requires temporary storage, mount an
explicit size-limited `/tmp`; no application persistence is required.

Inject/rotate both service credentials from the platform secret manager. TLS
termination and network policy ownership remain deployment-owner decisions;
provider-specific manifests are intentionally absent until a provider is
selected.

Dashboards use only the stable request/failure/stage/inference categories,
durations, candidate/result counts, readiness, and frozen policy/compatibility
versions. Alert on sustained not-ready, inference availability/timeout,
contract/package/key mismatch, unsafe-template, overload, and latency-budget
breaches. Never label metrics by correlation/candidate/model keys, payload
facts, explanation, or origin.

Release evidence retains the image digest/SBOM/scan summaries, contract and
fixture checksums, policy tuple, sanitized test/performance results, and
rollback drill. See the service `runbook.md` for outage and Backend fallback
procedures.

The local UAT matrix and release evidence index are under
`docs/testing/recommendation-agent-uat-matrix.md` and
`docs/testing/recommendation-agent-evidence/README.md`. A local PASS does not
authorize production routing. Backend owner approval, staged private-boundary
evidence, an immutable scanned image digest, alert review, and both rollback
drills are mandatory external gates.
