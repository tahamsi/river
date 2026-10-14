### Table 16. Computational cost

Computational cost and telemetry.

| comparison_mode | baseline | n | latency_ms | latency_ms_ci_low | latency_ms_ci_high | revision_latency_ms | revision_latency_ms_ci_low | revision_latency_ms_ci_high | tokens | tokens_ci_low | tokens_ci_high | model_calls | model_calls_ci_low | model_calls_ci_high | nli_calls | nli_calls_ci_low | nli_calls_ci_high | peak_memory_mb | peak_memory_mb_ci_low | peak_memory_mb_ci_high | estimated_cost | estimated_cost_ci_low | estimated_cost_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| controlled | naive_rag | 248 | 653.278 | 618.397 | 691.156 | **0.035** | 0.034 | 0.037 | 252.992 | 243.714 | 263.351 | 1.000 | 1.000 | 1.000 | **0.000** | 0.000 | 0.000 | 0.002 | 0.002 | 0.002 | **0.000** | 0.000 | 0.000 |
| controlled | stateless_rag | 248 | 604.676 | 572.191 | 639.282 | 0.037 | 0.036 | 0.039 | 252.992 | 243.714 | 263.351 | 1.000 | 1.000 | 1.000 | **0.000** | 0.000 | 0.000 | 0.002 | 0.002 | 0.002 | **0.000** | 0.000 | 0.000 |
| controlled | last_write_wins | 248 | 555.841 | 519.235 | 594.766 | 0.053 | 0.048 | 0.062 | 300.391 | 291.984 | 309.036 | 1.000 | 1.000 | 1.000 | **0.000** | 0.000 | 0.000 | 0.002 | 0.001 | 0.002 | **0.000** | 0.000 | 0.000 |
| controlled | recency_weighted | 248 | 566.398 | 531.884 | 602.082 | 0.069 | 0.056 | 0.089 | 300.391 | 291.984 | 309.036 | 1.000 | 1.000 | 1.000 | **0.000** | 0.000 | 0.000 | **0.002** | 0.001 | 0.002 | **0.000** | 0.000 | 0.000 |
| controlled | minicheck_no_revision | 248 | **55.468** | 48.619 | 66.971 | 55.135 | 48.304 | 66.614 | **0.000** | 0.000 | 0.000 | **0.000** | 0.000 | 0.000 | 1.117 | 1.028 | 1.210 | 0.051 | 0.034 | 0.084 | NA | NA | NA |
| controlled | river_d | 248 | 371.130 | 321.507 | 422.453 | 0.289 | 0.282 | 0.296 | 175.226 | 152.468 | 198.000 | 0.581 | 0.508 | 0.657 | **0.000** | 0.000 | 0.000 | 0.008 | 0.007 | 0.008 | **0.000** | 0.000 | 0.000 |
| controlled | river_l | 1240 | 400.275 | 362.502 | 437.653 | 21.866 | 20.809 | 22.993 | 186.896 | 167.961 | 205.515 | 0.635 | 0.569 | 0.700 | **0.000** | 0.000 | 0.000 | 0.005 | 0.005 | 0.005 | **0.000** | 0.000 | 0.000 |

- Unavailable telemetry is reported as missing rather than zero.
- estimated_cost is not applicable for controlled / minicheck_no_revision: it needs recorded generation cost metadata, which this system run did not produce (for example, a --no-generation run).
