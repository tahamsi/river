### Table 3. Paired recency decomposition

Arrival-order sensitivity decomposed on paired order-reversal examples.

| baseline | n | rfr | rfr_ci_low | rfr_ci_high | prs_jsd | prs_jsd_ci_low | prs_jsd_ci_high | order_consistency | order_consistency_ci_low | order_consistency_ci_high | stable_correct | stable_correct_ci_low | stable_correct_ci_high | stable_wrong | stable_wrong_ci_low | stable_wrong_ci_high | flip_to_correct | flip_to_correct_ci_low | flip_to_correct_ci_high | flip_to_wrong | flip_to_wrong_ci_low | flip_to_wrong_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| naive_rag | 58 | 0.095 | 0.000 | 0.238 | NA | NA | NA | 0.905 | 0.762 | 1.000 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| stateless_rag | 58 | 0.095 | 0.000 | 0.238 | NA | NA | NA | 0.905 | 0.762 | 1.000 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| last_write_wins | 58 | 1.000 | 1.000 | 1.000 | NA | NA | NA | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **0.000** | 0.000 | 0.000 | **0.931** | 0.828 | 1.000 | 0.034 | 0.000 | 0.103 |
| recency_weighted | 58 | 1.000 | 1.000 | 1.000 | NA | NA | NA | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **0.000** | 0.000 | 0.000 | **0.931** | 0.828 | 1.000 | 0.034 | 0.000 | 0.103 |
| minicheck_no_revision | 58 | **0.000** | 0.000 | 0.000 | NA | NA | NA | **1.000** | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | **0.000** | 0.000 | 0.000 |
| river_d | 58 | **0.000** | 0.000 | 0.000 | NA | NA | NA | **1.000** | 1.000 | 1.000 | 0.069 | 0.000 | 0.172 | 0.931 | 0.828 | 1.000 | 0.000 | 0.000 | 0.000 | **0.000** | 0.000 | 0.000 |
| river_l | 290 | **0.000** | 0.000 | 0.000 | **0.000** | 0.000 | 0.000 | **1.000** | 1.000 | 1.000 | **0.552** | 0.407 | 0.697 | 0.448 | 0.303 | 0.593 | 0.000 | 0.000 | 0.000 | **0.000** | 0.000 | 0.000 |

- Pair IDs are mandatory; rows without complete pairs are rejected.
- flip_to_correct is not applicable for naive_rag, stateless_rag: it needs reversed-order pairs scored by revision correctness, and this system emits no revision state.
- flip_to_wrong is not applicable for naive_rag, stateless_rag: it needs reversed-order pairs scored by revision correctness, and this system emits no revision state.
- prs_jsd is not applicable for last_write_wins, minicheck_no_revision, naive_rag, recency_weighted, river_d, stateless_rag: it needs a distribution over revision states, which this system does not emit.
- stable_correct is not applicable for naive_rag, stateless_rag: it needs reversed-order pairs scored by revision correctness, and this system emits no revision state.
- stable_wrong is not applicable for naive_rag, stateless_rag: it needs reversed-order pairs scored by revision correctness, and this system emits no revision state.
