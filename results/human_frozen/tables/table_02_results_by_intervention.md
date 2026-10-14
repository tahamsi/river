### Table 2. Results by intervention

Per-intervention revision accuracy and coverage.

| intervention | baseline | n | accuracy | accuracy_ci_low | accuracy_ci_high | coverage | coverage_ci_low | coverage_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| concurrent_disagreement | naive_rag | 16 | NA | NA | NA | NA | NA | NA |
| concurrent_disagreement | stateless_rag | 16 | NA | NA | NA | NA | NA | NA |
| concurrent_disagreement | last_write_wins | 16 | 0.000 | 0.000 | 0.000 | **1.000** | 1.000 | 1.000 |
| concurrent_disagreement | recency_weighted | 16 | 0.000 | 0.000 | 0.000 | **1.000** | 1.000 | 1.000 |
| concurrent_disagreement | minicheck_no_revision | 16 | 0.000 | 0.000 | 0.000 | 0.250 | 0.000 | 0.625 |
| concurrent_disagreement | river_d | 16 | 0.000 | 0.000 | 0.000 | 0.750 | 0.375 | 1.000 |
| concurrent_disagreement | river_l | 80 | 0.900 | 0.800 | 1.000 | 0.975 | 0.925 | 1.000 |
| evidence_removal | naive_rag | 66 | NA | NA | NA | NA | NA | NA |
| evidence_removal | stateless_rag | 66 | NA | NA | NA | NA | NA | NA |
| evidence_removal | last_write_wins | 66 | 0.500 | 0.500 | 0.500 | **1.000** | 1.000 | 1.000 |
| evidence_removal | recency_weighted | 66 | 0.500 | 0.500 | 0.500 | **1.000** | 1.000 | 1.000 |
| evidence_removal | minicheck_no_revision | 66 | 0.470 | 0.424 | 0.500 | 0.470 | 0.424 | 0.500 |
| evidence_removal | river_d | 66 | 0.970 | 0.924 | 1.000 | 0.500 | 0.500 | 0.500 |
| evidence_removal | river_l | 330 | 0.967 | 0.921 | 1.000 | 0.503 | 0.500 | 0.509 |
| expired_evidence | naive_rag | 52 | NA | NA | NA | NA | NA | NA |
| expired_evidence | stateless_rag | 52 | NA | NA | NA | NA | NA | NA |
| expired_evidence | last_write_wins | 52 | 0.500 | 0.500 | 0.500 | **1.000** | 1.000 | 1.000 |
| expired_evidence | recency_weighted | 52 | 0.500 | 0.500 | 0.500 | **1.000** | 1.000 | 1.000 |
| expired_evidence | minicheck_no_revision | 52 | 0.000 | 0.000 | 0.000 | **1.000** | 1.000 | 1.000 |
| expired_evidence | river_d | 52 | **1.000** | 1.000 | 1.000 | **1.000** | 1.000 | 1.000 |
| expired_evidence | river_l | 260 | **1.000** | 1.000 | 1.000 | **1.000** | 1.000 | 1.000 |
| insufficient_evidence_nei | naive_rag | 24 | NA | NA | NA | NA | NA | NA |
| insufficient_evidence_nei | stateless_rag | 24 | NA | NA | NA | NA | NA | NA |
| insufficient_evidence_nei | last_write_wins | 24 | 0.000 | 0.000 | 0.000 | **1.000** | 1.000 | 1.000 |
| insufficient_evidence_nei | recency_weighted | 24 | 0.000 | 0.000 | 0.000 | **1.000** | 1.000 | 1.000 |
| insufficient_evidence_nei | minicheck_no_revision | 24 | 0.583 | 0.333 | 0.833 | 0.417 | 0.167 | 0.667 |
| insufficient_evidence_nei | river_d | 24 | 0.083 | 0.000 | 0.250 | 0.917 | 0.750 | 1.000 |
| insufficient_evidence_nei | river_l | 120 | 0.917 | 0.750 | 1.000 | 0.083 | 0.000 | 0.250 |
| order_reversal | naive_rag | 58 | NA | NA | NA | NA | NA | NA |
| order_reversal | stateless_rag | 58 | NA | NA | NA | NA | NA | NA |
| order_reversal | last_write_wins | 58 | 0.483 | 0.448 | 0.500 | **1.000** | 1.000 | 1.000 |
| order_reversal | recency_weighted | 58 | 0.483 | 0.448 | 0.500 | **1.000** | 1.000 | 1.000 |
| order_reversal | minicheck_no_revision | 58 | 0.000 | 0.000 | 0.000 | 0.828 | 0.690 | 0.966 |
| order_reversal | river_d | 58 | 0.069 | 0.000 | 0.172 | 0.069 | 0.000 | 0.172 |
| order_reversal | river_l | 290 | 0.552 | 0.407 | 0.697 | 0.903 | 0.800 | 0.986 |
| repeated_weak_assertion | naive_rag | 10 | NA | NA | NA | NA | NA | NA |
| repeated_weak_assertion | stateless_rag | 10 | NA | NA | NA | NA | NA | NA |
| repeated_weak_assertion | last_write_wins | 10 | 0.300 | 0.100 | 0.500 | **1.000** | 1.000 | 1.000 |
| repeated_weak_assertion | recency_weighted | 10 | 0.300 | 0.100 | 0.500 | **1.000** | 1.000 | 1.000 |
| repeated_weak_assertion | minicheck_no_revision | 10 | 0.200 | 0.000 | 0.600 | 0.400 | 0.000 | 0.800 |
| repeated_weak_assertion | river_d | 10 | 0.400 | 0.000 | 0.800 | 0.800 | 0.400 | 1.000 |
| repeated_weak_assertion | river_l | 50 | 0.300 | 0.020 | 0.580 | 0.980 | 0.940 | 1.000 |
| retroactive_correction | naive_rag | 2 | NA | NA | NA | NA | NA | NA |
| retroactive_correction | stateless_rag | 2 | NA | NA | NA | NA | NA | NA |
| retroactive_correction | last_write_wins | 2 | 0.500 | 0.500 | 0.500 | **1.000** | 1.000 | 1.000 |
| retroactive_correction | recency_weighted | 2 | 0.500 | 0.500 | 0.500 | **1.000** | 1.000 | 1.000 |
| retroactive_correction | minicheck_no_revision | 2 | 0.000 | 0.000 | 0.000 | **1.000** | 1.000 | 1.000 |
| retroactive_correction | river_d | 2 | 0.500 | 0.500 | 0.500 | **1.000** | 1.000 | 1.000 |
| retroactive_correction | river_l | 10 | 0.400 | 0.400 | 0.400 | 0.900 | 0.900 | 0.900 |
| supported_genuine_update | naive_rag | 6 | NA | NA | NA | NA | NA | NA |
| supported_genuine_update | stateless_rag | 6 | NA | NA | NA | NA | NA | NA |
| supported_genuine_update | last_write_wins | 6 | 0.500 | 0.500 | 0.500 | **1.000** | 1.000 | 1.000 |
| supported_genuine_update | recency_weighted | 6 | 0.500 | 0.500 | 0.500 | **1.000** | 1.000 | 1.000 |
| supported_genuine_update | minicheck_no_revision | 6 | 0.000 | 0.000 | 0.000 | **1.000** | 1.000 | 1.000 |
| supported_genuine_update | river_d | 6 | 0.667 | 0.500 | 1.000 | **1.000** | 1.000 | 1.000 |
| supported_genuine_update | river_l | 30 | 0.500 | 0.500 | 0.500 | **1.000** | 1.000 | 1.000 |
| temporal_evolution | naive_rag | 6 | NA | NA | NA | NA | NA | NA |
| temporal_evolution | stateless_rag | 6 | NA | NA | NA | NA | NA | NA |
| temporal_evolution | last_write_wins | 6 | 0.000 | 0.000 | 0.000 | **1.000** | 1.000 | 1.000 |
| temporal_evolution | recency_weighted | 6 | 0.000 | 0.000 | 0.000 | **1.000** | 1.000 | 1.000 |
| temporal_evolution | minicheck_no_revision | 6 | 0.000 | 0.000 | 0.000 | **1.000** | 1.000 | 1.000 |
| temporal_evolution | river_d | 6 | 0.000 | 0.000 | 0.000 | 0.667 | 0.000 | 1.000 |
| temporal_evolution | river_l | 30 | 0.133 | 0.000 | 0.400 | **1.000** | 1.000 | 1.000 |
| unsupported_late_assertion | naive_rag | 8 | NA | NA | NA | NA | NA | NA |
| unsupported_late_assertion | stateless_rag | 8 | NA | NA | NA | NA | NA | NA |
| unsupported_late_assertion | last_write_wins | 8 | 0.375 | 0.125 | 0.500 | **1.000** | 1.000 | 1.000 |
| unsupported_late_assertion | recency_weighted | 8 | 0.375 | 0.125 | 0.500 | **1.000** | 1.000 | 1.000 |
| unsupported_late_assertion | minicheck_no_revision | 8 | 0.250 | 0.000 | 0.750 | 0.250 | 0.000 | 0.750 |
| unsupported_late_assertion | river_d | 8 | 0.250 | 0.000 | 0.750 | 0.500 | 0.000 | 1.000 |
| unsupported_late_assertion | river_l | 40 | 0.175 | 0.000 | 0.375 | 0.950 | 0.850 | 1.000 |

- Each intervention row is computed only from examples carrying that intervention label.
- accuracy is not applicable for 20 of 70 rows: it needs a predicted revision state, which this system does not emit.
- coverage is not applicable for 20 of 70 rows: it needs a predicted revision state, which this system does not emit.
