### Table 1. Overall controlled baseline comparison

Overall controlled comparison across normalized RIVER-Bench prediction records.

| baseline | n | answer_accuracy | answer_accuracy_ci_low | answer_accuracy_ci_high | rfr | rfr_ci_low | rfr_ci_high | usr | usr_ci_low | usr_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| naive_rag | 246 | 0.686 | 0.603 | 0.767 | 0.432 | 0.286 | 0.579 | 0.333 | 0.000 | 1.000 |
| stateless_rag | 246 | 0.686 | 0.603 | 0.767 | 0.432 | 0.286 | 0.579 | 0.333 | 0.000 | 1.000 |
| last_write_wins | 246 | 0.622 | 0.587 | 0.662 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| recency_weighted | 246 | 0.622 | 0.587 | 0.662 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| minicheck_no_revision | 246 | NA | NA | NA | **0.000** | 0.000 | 0.000 | NA | NA | NA |
| river_d | 246 | 0.628 | 0.524 | 0.732 | **0.000** | 0.000 | 0.000 | **0.000** | 0.000 | 0.000 |
| river_l | 1230 | **0.801** | 0.725 | 0.874 | **0.000** | 0.000 | 0.000 | 0.667 | 0.000 | 1.000 |

- Controlled rows share evidence and budgets where technically valid.
- Lower is better for RFR, USR, and ECE.
- answer_accuracy is not applicable for minicheck_no_revision: it needs strict direct-answer labels produced under generation contract v2, which this system does not emit.
- usr is not applicable for minicheck_no_revision: it needs a selected hypothesis this system named on at least one late-assertion row, which this system does not emit.
