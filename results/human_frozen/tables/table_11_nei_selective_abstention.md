### Table 11. NEI and selective abstention

Selective abstention on NEI and evidence-removal examples.

| baseline | n | precision | precision_ci_low | precision_ci_high | recall | recall_ci_low | recall_ci_high | f1 | f1_ci_low | f1_ci_high | coverage | coverage_ci_low | coverage_ci_high | selective_risk | selective_risk_ci_low | selective_risk_ci_high | aurc | aurc_ci_low | aurc_ci_high | acc_at_80cov | acc_at_80cov_ci_low | acc_at_80cov_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| naive_rag | 90 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| stateless_rag | 90 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| last_write_wins | 90 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **1.000** | 1.000 | 1.000 | 0.633 | 0.567 | 0.700 | NA | NA | NA | NA | NA | NA |
| recency_weighted | 90 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **1.000** | 1.000 | 1.000 | 0.633 | 0.567 | 0.700 | NA | NA | NA | NA | NA | NA |
| minicheck_no_revision | 90 | 0.918 | 0.837 | 0.981 | 0.818 | 0.673 | 0.960 | 0.865 | 0.768 | 0.944 | 0.456 | 0.378 | 0.533 | 1.000 | 1.000 | 1.000 | NA | NA | NA | NA | NA | NA |
| river_d | 90 | 0.943 | 0.853 | 1.000 | 0.600 | 0.443 | 0.769 | 0.733 | 0.598 | 0.854 | 0.611 | 0.544 | 0.678 | 0.400 | 0.235 | 0.557 | 0.364 | 0.168 | 0.610 | 0.614 | 0.420 | 0.795 |
| river_l | 450 | **0.964** | 0.904 | 1.000 | **0.960** | 0.882 | 1.000 | **0.962** | 0.911 | 0.998 | 0.391 | 0.322 | 0.458 | **0.062** | 0.000 | 0.181 | **0.007** | 0.000 | 0.017 | **0.983** | 0.961 | 1.000 |

- Risk-coverage metrics require confidence values and correctness labels.
- acc_at_80cov is not applicable for last_write_wins, minicheck_no_revision, naive_rag, recency_weighted, stateless_rag: it needs a confidence score, which this system does not emit.
- aurc is not applicable for last_write_wins, minicheck_no_revision, naive_rag, recency_weighted, stateless_rag: it needs a confidence score, which this system does not emit.
- coverage is not applicable for naive_rag, stateless_rag: it needs a predicted revision state, which this system does not emit.
- f1 is not applicable for naive_rag, stateless_rag: it needs a predicted revision state, which this system does not emit.
- precision is not applicable for naive_rag, stateless_rag: it needs a predicted revision state, which this system does not emit.
- recall is not applicable for naive_rag, stateless_rag: it needs a predicted revision state, which this system does not emit.
- selective_risk is not applicable for naive_rag, stateless_rag: it needs a predicted revision state, which this system does not emit.
