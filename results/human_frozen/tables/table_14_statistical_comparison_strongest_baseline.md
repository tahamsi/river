### Table 14. Statistical comparison with strongest baseline

Paired statistical comparison against the strongest non-RIVER baseline.

| candidate | reference | metric | estimate | ci_low | ci_high | mcnemar_p | mcnemar_p_holm | permutation_p | permutation_p_holm | cohen_dz | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| river_d | last_write_wins | revision_accuracy_difference | 0.146 | 0.069 | 0.224 | 0.000 | 0.000 | 0.000 | 0.000 | 0.243 | 246 |
| river_l | last_write_wins | revision_accuracy_difference | **0.386** | 0.309 | 0.459 | **0.000** | **0.000** | **0.000** | **0.000** | **0.636** | 246 |

- The strongest baseline is selected by answer-or-revision accuracy, excluding RIVER variants.
