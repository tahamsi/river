# Primary run (2026-09-21)

The GPU run that produced the structured decisions behind the reported results.
Its rendered answers and tables are superseded by
[`results/human_frozen/`](../../results/human_frozen/), which kept every
structured decision unchanged and reran only rendering, answer scoring,
MiniCheck, the audit and the tables. The ten structured prediction files from
this run (five single-seed baselines and five RIVER-L seeds) are archived in
`results/human_frozen/inputs/`.

| File | Contents |
| --- | --- |
| `structured_results.json` | State, active-selection and joint accuracy for each run, with Wilson intervals, from `scripts/release/summarize_structured_rerun.py` |
| `prediction_validation.txt` | Row counts for the ten prediction files |
| `experiment_environment.txt`, `start_utc.txt`, `finish_utc.txt`, `primary_rerun.exit_code` | Run environment and timing |

The run exited with code 2 because the table step required MiniCheck output
that had not yet been produced. Its tables are not used for reporting. The
reasons for the final release are set out in
[`docs/release/RELEASE_NOTES.md`](../../docs/release/RELEASE_NOTES.md).
