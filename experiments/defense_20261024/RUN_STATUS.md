# Defense experiments: launch status

Observed: **24 October 2026, 23:46 MSK**. This is a launch record, not a completed-results report.

## Location and ownership

- SSH account: `kenshyo_local`, user `aarzumanyan`.
- Isolated root: `/home/aarzumanyan/decovae_defense_20261024`.
- Source account `kenshyo_local_ilya` was used read-only. No training, evaluator, installation, or modification was performed there.
- Transferred reference: 18,127 files, 3,407,503,330 bytes; SHA256 verification passed. A small local FFT-only evidence bundle was added separately with its own hashes.
- Reference CSV/checkpoint numerical reproduction: maximum error 8.33e-17 (real) and 5.55e-17 (volatility), 32 windows and 128 predictive draws each.

## Running and queued

Two H1 trainings are currently running, with no failed jobs. Twelve new trainings are planned in total:

- Real data: noise encoder Zn and volatility-feature B4, seeds 3–4 (four trainings).
- Strong-volatility data: Dmc1, Zn, and B4, seeds 3–4 (six trainings).
- Real Dmc1 control: seeds 3–4, added to current-environment `srv14r_Dmc1_w0.2` seeds 0–2 (two trainings).

The published five-seed `srv10_Dmc1` used a different training environment. Its comparison is retained as historical evidence; the primary H1 analysis uses the matched current-environment control. No training architecture, loss or data protocol was altered.

After training, the queue evaluates the checkpoints, constructs paired seed/series intervals for CRPS, variance errors and coverage/width differences, and runs frozen-checkpoint synthetic diagnostics (three seeds). These distinguish reconstruction from forecast, clean-mean accuracy from uncertainty, and simple seasonality from negative/no-season cases.

The optional external reference reruns Chronos-Bolt-small and evaluates it alongside our frozen models with exactly the same nine-quantile score and seasonal-naive reference. Potential pretraining overlap remains a limitation. Main MC128 CRPS and qCRPS9 are not pooled.

Completed LOSO, source concentration, period-feature controls, scaling/KernelSynth, calibration, and FFT-only ablation evidence are reused rather than retrained.

## Review coverage

| Review topic | Evidence / remaining work |
|---|---|
| Source concentration and uncertainty | Existing per-source and completed 15-source LOSO; report includes source bootstrap intervals. |
| Period-estimator input control | Existing N6s; distinguish estimator information from hard additive decomposition. |
| Development exposure | Disclose inspected OOD sources; do not call them untouched. |
| Estimator failure / fallback | Existing interventions and hybrid gating; new no-season/weak-season synthetic diagnostics. |
| Original seasonal decoder | Existing controlled FFT-only study, three variants, three horizons including 256, three seeds. |
| Final H1, more seeds | New five-seed matched extension running/queued. |
| External comparison | New matched-score reference queued; clean zero-shot external comparison remains unresolved. |
| Reproduction | Hash-verified private snapshot and reproduction checks completed. Public tag/release is not published. |

## Shared-machine controls

Only registered child process groups may be signalled, after UID/PGID/session/start-time verification. No foreign process control or system-wide tuning. Children use one numerical-library thread, nice 10, efficiency-core affinity, and maximum two concurrent jobs. The initial queue is capped at 24 GiB RAM / four CPU equivalents; follow-up queue at 16 GiB / two equivalents. Swap is disabled for these services only.

Telemetry is sampled every 15 seconds. Foreign GPU work, high temperatures, resource pressure, or monitoring failure causes only our work to checkpoint/stop; it resumes when safe. Eight guard tests passed. Numeric sanity checks for the shared quantile score and paired bootstrap passed.

At 23:46: CPU 50 C, GPU 49 C, 4,786 / 24,564 MiB GPU memory, 115.85 GiB RAM available, zero CPU-pressure average and no foreign GPU compute PIDs. GPU utilization is high while idle capacity is available; it is released when foreign GPU work appears.

Own user lingering is enabled so the services survive logout. Services:

- `decovae-defense-20261024.service`: initial H1 queue, cutoff 25 October 20:00 MSK.
- `decovae-defense-analysis-20261024.service`: matched controls, diagnostics and report assembly, cutoff 25 October 23:00 MSK.

## Files to inspect next

Under the remote root:

- `ops/queue_status.json`, `ops/queue_events.jsonl`, `ops/telemetry.jsonl`: live state and resource history.
- `ops/h1_queue_summary.json`: archived first-queue result after handoff.
- `final_results/H1_FINDINGS.md`, `h1_comparisons.csv`, `matched_h1_complete.json`: primary H1 result only when the matched marker passes.
- `final_results/component_intervals.csv`, `COMPONENT_PROTOCOL.md`, `components_complete.json`: component/interval diagnostics.
- `final_results/EXTERNAL_REFERENCE.md`, `external_qcrps_complete.json`: matched external reference, if successful.
- `final_results/DEFENSE_STATUS.md`, `REVIEW_RESPONSE_DRAFT.md`, `evidence_complete.json`: assembled report packet. Availability flags disclose incomplete diagnostics.

Operations sources and deployment revisions are preserved. The queue does not publish a Git tag, push changes, or modify the final presentation. The current report/presentation paths are still needed for the final editing pass.
