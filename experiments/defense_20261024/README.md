# Final H1 validation, 24 October 2026

This extension reuses the final-model source, cached windows, and frozen reference
runs from the read-only `kenshyo_local_ilya` account. New work runs exclusively as
`aarzumanyan` in `/home/aarzumanyan/decovae_defense_20261024`.

## Scientific protocol

Add training seeds 3 and 4 to the existing seeds 0, 1, and 2:

- Real cache: `Dmc1Zn` and `Dmc1B4`; the published `Dmc1` baseline has five seeds but is a historical control.
- Volatility cache: `Dmc1`, `Dmc1Zn`, and `Dmc1B4`.

The initial queue contains ten new trainings. Architectures and objectives are unchanged. B4 uses
handcrafted residual-volatility statistics; it is not a capacity-matched raw-input
encoder. In `h1fm_strong`, **test_id has strong volatility** and **test_ood has the
base profile**. Keep both splits and the first-32-step results distinct.

Batch 256, AdamW 0.001, gradient clipping 1, reconstruction weight 0.2, KL weight
0.01, context 512, horizon 96, 8 training mixture components, 128 evaluation draws.
Use the original 472-epoch budget, validation every 8 epochs, patience 10, and
`--fast-val`. Select checkpoints exclusively using validation CRPS.

Before training, reproduce the saved scores of one real and one volatility
checkpoint. All copied bytes have a transfer manifest and a SHA256 check.

## Shared-machine controls

- Only explicitly registered child process groups can receive signals. UID,
  process-group ID, session ID, and Linux process start time must still match.
- No `pkill`, GPU clock/power changes, or modification of another user's files.
- Start with one job; allow at most two. Pin children to efficiency CPUs 16–19
  and 20–23, use nice 10 and one numerical-library thread.
- Stop/checkpoint our jobs when another GPU workload appears, monitoring fails,
  CPU temperature reaches 94 C, GPU temperature reaches 82 C, available RAM drops
  below 24 GiB, or disk reserve drops below 100 GiB. Resume at lower thresholds.
- The user systemd service additionally caps the whole queue to 24 GiB RAM,
  no swap, and four CPU equivalents. No system-wide limits are changed.
- Write telemetry every 15 seconds: temperatures, GPU memory/utilization,
  CPU pressure/load, selected CPU frequencies and utilization, and owned PIDs.

`ops/control.json` controls the concurrency (`max_parallel`, clamped to 0–2).
SIGTERM to the supervisor requests checkpointing of its children. Interruption
markers prevent an incomplete evaluation being mistaken for a finished job.

## Outputs

New runs have the prefix `defense_h1_`; reference runs retain their original names.
The plan and full source hashes are in `ops/plan.json`; state, events, and telemetry
are in `ops/queue_status.json`, `ops/queue_events.jsonl`, and `ops/telemetry.jsonl`.
No test score is used for checkpoint or hyperparameter selection.


## Queued follow-up diagnostics

After the initial H1 queue releases its lock, the separate owned analysis service first adds real-data Dmc1 seeds 3 and 4 to the current-environment `srv14r_Dmc1_w0.2` seeds 0–2. This makes twelve new trainings in total. The primary H1 result uses this matched control; the initial historical-control summary is archived separately. The service then runs:

1. Frozen lab checkpoints, three seeds, 128 draws: component recovery on context
   and forecast horizon; clean-mean MSE, CRPS, interval coverage and widths.
   Added controls include no seasonality, weak seasonality, two frequencies,
   and a slow trend wave plus a fast seasonal wave. No truth enters model inputs.
2. Chronos-Bolt-small on the same frozen real windows, followed by a matched
   nine-quantile score for Chronos, seasonal-naive, PD, Transformer, and the
   period-feature Transformer. The model revision is recorded. Potential
   pretraining overlap remains a limitation; this is an external reference.
3. A results packet with source-level LOSO intervals, final five-seed H1,
   synthetic diagnostics, calibration, figures, and an English review-response
   draft. Optional diagnostic failures are reported rather than preventing the
   packet from being assembled.

The H1 service deadline is 25 October 20:00 MSK; follow-up analysis ends by
25 October 23:00 MSK. These are compute cutoffs, leaving Monday morning for
review of the report and presentation before the 12:30 defense deadline.
The services persist after SSH disconnects. No public release or push is done.

All completion flags require successful execution and nonempty outputs. Source
hashes, checkpoint hashes, immutable transferred references, and resumable
fragments distinguish completed results from partially written files.
