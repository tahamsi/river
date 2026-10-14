### Table 9. Conflict detection with negative controls

Conflict detection including negative controls.

| baseline | n | precision | precision_ci_low | precision_ci_high | recall | recall_ci_low | recall_ci_high | macro_f1 | macro_f1_ci_low | macro_f1_ci_high | specificity | specificity_ci_low | specificity_ci_high | false_conflict | false_conflict_ci_low | false_conflict_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| naive_rag | 88 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| stateless_rag | 88 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| last_write_wins | 88 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.450 | 0.413 | 0.482 | **1.000** | 1.000 | 1.000 | **0.000** | 0.000 | 0.000 |
| recency_weighted | 88 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.450 | 0.413 | 0.482 | **1.000** | 1.000 | 1.000 | **0.000** | 0.000 | 0.000 |
| minicheck_no_revision | 88 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.450 | 0.413 | 0.482 | **1.000** | 1.000 | 1.000 | **0.000** | 0.000 | 0.000 |
| river_d | 88 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.443 | 0.397 | 0.476 | 0.972 | 0.909 | 1.000 | 0.028 | 0.000 | 0.091 |
| river_l | 440 | **0.720** | 0.424 | 0.960 | **0.900** | 0.800 | 1.000 | **0.874** | 0.745 | 0.960 | 0.922 | 0.835 | 0.991 | 0.078 | 0.009 | 0.165 |

- Temporal-evolution and non-conflict interventions serve as negative controls.
- false_conflict is not applicable for naive_rag, stateless_rag: it needs a predicted revision state, which this system does not emit.
- macro_f1 is not applicable for naive_rag, stateless_rag: it needs a predicted revision state, which this system does not emit.
- precision is not applicable for naive_rag, stateless_rag: it needs a predicted revision state, which this system does not emit.
- recall is not applicable for naive_rag, stateless_rag: it needs a predicted revision state, which this system does not emit.
- specificity is not applicable for naive_rag, stateless_rag: it needs a predicted revision state, which this system does not emit.
