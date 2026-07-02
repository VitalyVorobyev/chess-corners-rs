# chess-corners-rs — internal knowledge base

This directory is the **internal** knowledge base for the workspace: planning,
design rationale, decisions, and reference notes. It is *not* published — the
public docs are the [book](../book/) (`book/src/`), the rustdoc API reference,
the README, and the CHANGELOG.

> **Internal vs published.** Because `docs/` is not published, design
> *rationale* and trade-off history are welcome here. The `CLAUDE.md`
> "no dev-history on public surfaces" rule binds only the book, README,
> rustdoc, and CHANGELOG — not these files.

## Start here

| If you want… | Read |
|--------------|------|
| The plan & milestones (incl. `v1.0.0`) | [`ROADMAP.md`](ROADMAP.md) |
| The task list | [`BACKLOG.md`](BACKLOG.md) |
| A map of every algorithm and how they connect | [`design/algorithms-index.md`](design/algorithms-index.md) |
| How the crates are layered | [`design/architecture.md`](design/architecture.md) |

## Layout

```
docs/
  README.md          ← you are here (KB index)
  ROADMAP.md         ← milestones + sequencing
  BACKLOG.md         ← task registry (IDs, priority, status, deps)
  design/            ← architecture + per-workstream design docs / RFCs
    architecture.md          crate layering & data flow
    algorithms-index.md      atomic-algorithm map + DAG
    api-v1.0.md              public API stabilization RFC
    cpp-vcpkg-bindings.md    C++/vcpkg binding design
    site-architecture.md     GitHub Pages site design
    perf-profiling.md        performance method + plan
  reference/         ← current comparative notes
    detector-comparison.md   ChESS vs Radon
    refiner-comparison.md    refiner accuracy/throughput
  process/           ← how we work
    review-workflow.md       release review pipeline
    subagent-workflow.md     subagent dispatch guide
  decisions/         ← architecture decision records (ADRs)
  changelog/         ← per-version release notes (indexed by ../CHANGELOG.md)
```

## Conventions

- **Design docs** live in `design/<name>.md`. A design doc is the place for
  rationale, alternatives considered, and open questions; the corresponding
  work is tracked as `<WS>-NN` tasks in `BACKLOG.md`.
- **ADRs** in `decisions/` are numbered and immutable once accepted; supersede
  rather than edit.
- **Tasks** use `<WS>-NN` IDs (see the `BACKLOG.md` legend). Every backlog row
  carries a milestone and dependency list; every ROADMAP milestone lists the
  task IDs that close it. Keep the two in sync.
- **Changelog**: `[Unreleased]` stays inline in the root `CHANGELOG.md`;
  released notes move under `changelog/X.Y.Z.md`.

## Program status

Milestones M1–M6 are done; only the 1.0.0 release act (`API-09`) is deferred by
choice. See [`ROADMAP.md`](ROADMAP.md) for the milestone table and
[`BACKLOG.md`](BACKLOG.md) for the task registry.
