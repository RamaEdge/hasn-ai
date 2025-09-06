# HASN Implementation Issues (Detailed)

Each section below is a detailed issue plan ready to paste into Linear.  

---

## 0) Groundwork & Project Hygiene
**Context**: Make repo importable, deterministic, and testable.  
**Scope**: CI setup, logging, RNG seeding, Dockerfile.dev.

**Tasks**
- Add `pyproject.toml` + lock file (Poetry or pip-tools) with pinned versions.  
- Implement `src/common/logging.py` (structured JSON logs, INFO default, DEBUG override).  
- Implement RNG seeding utility (`src/common/random.py`) that seeds Python `random`, `numpy`, and torch if available.  
- Add `Dockerfile.dev` with hot-reload for API.  
- Create GitHub Actions pipeline: lint → test → build.  
- Create deterministic test with known input spike pattern and expected raster.

**Deliverables**
- CI passes on clean clone.  
- Running `pytest` green.  
- `uvicorn src.api.main:app` starts without missing modules.

**Out of Scope**
- Production-grade Dockerfile (done later under deployment).

---

## 1) Backend Abstraction & Minimal SNN Engine
**Context**: Abstract backend to allow swapping (NumPy/Norse).

**Tasks**
- Create `src/core/backend/interface.py` defining `BrainBackend`.  
- Implement `NumpyBackend`:  
  - Store neurons, synapses in numpy arrays.  
  - `step(inputs)` updates membrane potentials, spikes.  
- Add placeholder `NorseBackend` that raises `NotImplementedError` but satisfies interface.  
- Environment variable `HASN_BACKEND` chooses backend.  
- Unit tests for numpy backend.

**Deliverables**
- `BrainBackend` interface defined.  
- `NumpyBackend` works with 10-neuron toy example.  
- Switching backend via env var works.

**Out of Scope**
- Full STDP; handled later under Cognitive Layer.

---

## 2) Cognitive Layer (Layers + Gating)
**Context**: Implement sensory, associative, working, episodic, semantic, executive layers.

**Tasks**
- Define Pydantic models for `CognitiveConfig` and `EpisodeTrace`.  
- Implement Sensory Encoding: text→Poisson spikes, embeddings→spikes.  
- Associative memory: Hebbian updates between co-active spikes.  
- Working memory buffer with TTL eviction.  
- Episodic memory: store `EpisodeTrace` with timestamp.  
- Semantic memory: consolidation process merges traces.  
- Executive layer: arbitration logic (decides recall vs consolidation).  
- Tests for TTL eviction and consolidation.

**Deliverables**
- Example: encode “cat” → spikes → stored in episodic memory → consolidated into semantic after N exposures.  
- Unit tests confirm TTL eviction.

**Out of Scope**
- Advanced attention mechanisms (future enhancement).

---

## 3) Knowledge Store (Snapshots + Vector Index)
**Context**: Enable save/load portability + hybrid retrieval.

**Tasks**
- Define snapshot schema v1 (JSON).  
- Implement `BrainSerializer` to dump/load state.  
- Add save/load endpoints `/state/save`, `/state/load/{id}`.  
- Integrate Qdrant: collection `hasn_semantic_v1` with vectors + payload.  
- Implement `/knowledge/search` API: hybrid recall (spike sim + vector search).  
- Tests for save→load roundtrip and top-k retrieval.

**Deliverables**
- Running `POST /state/save` and then `/state/load` restores deterministic behavior.  
- Search returns consistent results.

**Out of Scope**
- Snapshot migration (handled in issue 9).

---

## 4) Ingestion & Continuous Training Pipeline
**Context**: Safe continual learning with governance.

**Tasks**
- Define `IngestItem` and `TrainingJob` models.  
- Implement `/ingest/submit`: validates license + robots.txt.  
- Create quarantine buffer (local folder or Redis list).  
- Implement deduplication (hash content).  
- Implement replay trainer: runs Hebbian updates for ingested items.  
- Implement consolidation API `/train/consolidate`.  
- Metrics endpoint `/train/metrics`.  
- Unit + integration tests.

**Deliverables**
- Submitting item → appears in quarantine → replay updates network.  
- Dedup works (duplicate skipped).  
- Metrics endpoint reports novelty, drift.

**Out of Scope**
- Human-in-the-loop GUI (manual approval can be stubbed).

---

## 5) Chat Interaction (Text)
**Context**: Primary UX; spike→text with provenance.

**Tasks**
- Define `ChatRequest` and `ChatResponse`.  
- Implement `POST /chat` using cognitive layer recall + response generator.  
- Include provenance in responses (episode IDs, semantic links).  
- Implement `/chat/{session}/context` and `/chat/{session}/reset`.  
- Add rate limiting middleware.  
- Add safety filter (PII/NSFW regex stub).  

**Deliverables**
- Posting “hello” returns valid `ChatResponse` JSON.  
- Provenance field populated.  
- Reset clears working memory.

**Out of Scope**
- Rich conversation history UI.

---

## 6) Vision Interaction
**Context**: Multimodal grounding via embeddings.

**Tasks**
- Define `VisionRequest`.  
- Implement `/vision/ingest` (store embedding in episodic memory).  
- Implement `/vision/query` (retrieve related episodic/semantic traces).  
- Add support for CLIP embedding input (client-side for now).  
- Test with toy embeddings.

**Deliverables**
- Ingesting an embedding allows recall through `/vision/query`.  
- End-to-end test passes with sample dataset.

**Out of Scope**
- Server-side CLIP integration (future GPU module).

---

## 7) Security & STRIDE Controls
**Context**: Align implementation with threat model.

**Tasks**
- Add API key middleware (fastapi).  
- Add JWT auth option.  
- Implement `AuditEvent` model + audit log on every state change.  
- Add snapshot integrity check (SHA-256).  
- Implement `/admin/audit` API.  
- Implement `/admin/keys` for key rotation.  

**Deliverables**
- Snapshot tamper detection works.  
- Audit log captures API calls.  

**Out of Scope**
- Full RBAC UI.

---

## 8) Observability (Tracing + Metrics + Rasters)
**Context**: Debugability and monitoring.

**Tasks**
- Add Prometheus `/metrics` endpoint with counters/gauges.  
- Metrics: spikes/sec, synapse updates, recall hit ratio.  
- Implement `/inspect/spikes?session_id` to export raster (JSON/PNG).  
- Grafana dashboards (local + cloud templates).  
- Tests: counters increment as expected.

**Deliverables**
- Grafana shows spikes/sec.  
- Raster export matches deterministic test.

**Out of Scope**
- Long-term telemetry storage (phase 2).

---

## 9) Serializer & Versioning
**Context**: Portability across versions.

**Tasks**
- Implement snapshot version tag.  
- Write migration function `migrate_snapshot(snapshot, to_version)`.  
- Implement `/state/migrate?to=v2`.  
- Tests with dummy v1→v2 upgrade.

**Deliverables**
- Loading v1 snapshot in v2 works.  
- Corruption detected and blocked.

**Out of Scope**
- v3+ schema (future work).

---

## 10) CLI & Tooling
**Context**: Developer + ops convenience.

**Tasks**
- Build `hasn` CLI (typer).  
- Commands:  
  - `hasn snapshot save|load`  
  - `hasn train replay|consolidate`  
  - `hasn inspect spikes`  
- Tests for CLI commands.

**Deliverables**
- CLI wraps APIs correctly.  
- CLI works offline with local state.

**Out of Scope**
- Full TUI/GUI.

---

## 11) Test Strategy & CI
**Context**: Reliable shipping.

**Tasks**
- Golden master tests: fixed input → expected raster checksum.  
- API contract tests with Pydantic.  
- Coverage reports integrated into CI.  
- GPU job for Norse backend (nightly).  
- Lint + mypy checks.

**Deliverables**
- >90% coverage core.  
- CI blocks merge if coverage drops.  
- Contract tests fail on schema drift.

**Out of Scope**
- Full fuzz testing.

---
