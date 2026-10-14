### Table 1. Revision-state comparison

Conflict detection and abstention among systems that decide a revision state.

| baseline | n | conflict_macro_f1 | conflict_macro_f1_ci_low | conflict_macro_f1_ci_high | abstention_f1 | abstention_f1_ci_low | abstention_f1_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| last_write_wins | 246 | 0.480 | 0.466 | 0.491 | 0.000 | 0.000 | 0.000 |
| recency_weighted | 246 | 0.480 | 0.466 | 0.491 | 0.000 | 0.000 | 0.000 |
| minicheck_no_revision | 246 | 0.480 | 0.466 | 0.491 | 0.681 | 0.567 | 0.785 |
| river_d | 246 | 0.524 | 0.469 | 0.635 | 0.407 | 0.299 | 0.519 |
| river_l | 1230 | **0.807** | 0.678 | 0.896 | **0.863** | 0.772 | 0.936 |

- Systems emitting no revision state are absent by construction, not by omission.
