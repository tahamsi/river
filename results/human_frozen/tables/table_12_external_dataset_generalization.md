### Table 12. External-dataset generalization

Generalization across external datasets.

| comparison_mode | source_dataset | baseline | n | accuracy | accuracy_ci_low | accuracy_ci_high | rfr | rfr_ci_low | rfr_ci_high | abstention_f1 | abstention_f1_ci_low | abstention_f1_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| controlled | 2wikimultihopqa | naive_rag | 58 | NA | NA | NA | 0.095 | 0.000 | 0.238 | NA | NA | NA |
| controlled | 2wikimultihopqa | stateless_rag | 58 | NA | NA | NA | 0.095 | 0.000 | 0.238 | NA | NA | NA |
| controlled | 2wikimultihopqa | last_write_wins | 58 | 0.483 | 0.448 | 0.500 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| controlled | 2wikimultihopqa | recency_weighted | 58 | 0.483 | 0.448 | 0.500 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| controlled | 2wikimultihopqa | minicheck_no_revision | 58 | 0.000 | 0.000 | 0.000 | **0.000** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| controlled | 2wikimultihopqa | river_d | 58 | 0.069 | 0.000 | 0.172 | **0.000** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| controlled | 2wikimultihopqa | river_l | 290 | 0.552 | 0.407 | 0.697 | **0.000** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| controlled | fever | naive_rag | 24 | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| controlled | fever | stateless_rag | 24 | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| controlled | fever | last_write_wins | 24 | 0.000 | 0.000 | 0.000 | NA | NA | NA | 0.000 | 0.000 | 0.000 |
| controlled | fever | recency_weighted | 24 | 0.000 | 0.000 | 0.000 | NA | NA | NA | 0.000 | 0.000 | 0.000 |
| controlled | fever | minicheck_no_revision | 24 | 0.583 | 0.333 | 0.833 | NA | NA | NA | 0.737 | 0.500 | 0.909 |
| controlled | fever | river_d | 24 | 0.083 | 0.000 | 0.250 | NA | NA | NA | 0.154 | 0.000 | 0.400 |
| controlled | fever | river_l | 120 | 0.917 | 0.750 | 1.000 | NA | NA | NA | 0.957 | 0.857 | 1.000 |
| controlled | longmemeval | naive_rag | 12 | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| controlled | longmemeval | stateless_rag | 12 | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| controlled | longmemeval | last_write_wins | 12 | 0.250 | 0.083 | 0.417 | NA | NA | NA | 0.000 | 0.000 | 0.000 |
| controlled | longmemeval | recency_weighted | 12 | 0.250 | 0.083 | 0.417 | NA | NA | NA | 0.000 | 0.000 | 0.000 |
| controlled | longmemeval | minicheck_no_revision | 12 | 0.000 | 0.000 | 0.000 | NA | NA | NA | 0.000 | 0.000 | 0.000 |
| controlled | longmemeval | river_d | 12 | 0.333 | 0.083 | 0.667 | NA | NA | NA | 0.000 | 0.000 | 0.000 |
| controlled | longmemeval | river_l | 60 | 0.317 | 0.150 | 0.483 | NA | NA | NA | 0.000 | 0.000 | 0.000 |
| controlled | pat_questions | naive_rag | 118 | NA | NA | NA | 0.739 | 0.545 | 0.909 | NA | NA | NA |
| controlled | pat_questions | stateless_rag | 118 | NA | NA | NA | 0.739 | 0.545 | 0.909 | NA | NA | NA |
| controlled | pat_questions | last_write_wins | 118 | 0.500 | 0.500 | 0.500 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| controlled | pat_questions | recency_weighted | 118 | 0.500 | 0.500 | 0.500 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| controlled | pat_questions | minicheck_no_revision | 118 | 0.263 | 0.195 | 0.322 | **0.000** | 0.000 | 0.000 | 0.939 | 0.877 | 0.986 |
| controlled | pat_questions | river_d | 118 | **0.983** | 0.958 | 1.000 | **0.000** | 0.000 | 0.000 | **0.969** | 0.915 | 1.000 |
| controlled | pat_questions | river_l | 590 | 0.981 | 0.956 | 1.000 | **0.000** | 0.000 | 0.000 | 0.966 | 0.913 | 1.000 |
| controlled | wikicontradict | naive_rag | 34 | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| controlled | wikicontradict | stateless_rag | 34 | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| controlled | wikicontradict | last_write_wins | 34 | 0.176 | 0.059 | 0.294 | NA | NA | NA | 0.000 | 0.000 | 0.000 |
| controlled | wikicontradict | recency_weighted | 34 | 0.176 | 0.059 | 0.294 | NA | NA | NA | 0.000 | 0.000 | 0.000 |
| controlled | wikicontradict | minicheck_no_revision | 34 | 0.118 | 0.000 | 0.294 | NA | NA | NA | 0.267 | 0.000 | 0.533 |
| controlled | wikicontradict | river_d | 34 | 0.176 | 0.000 | 0.353 | NA | NA | NA | 0.000 | 0.000 | 0.000 |
| controlled | wikicontradict | river_l | 170 | 0.553 | 0.359 | 0.741 | NA | NA | NA | 0.000 | 0.000 | 0.000 |
| controlled | wikipedia_revision | naive_rag | 2 | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| controlled | wikipedia_revision | stateless_rag | 2 | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| controlled | wikipedia_revision | last_write_wins | 2 | 0.500 | 0.500 | 0.500 | NA | NA | NA | 0.000 | 0.000 | 0.000 |
| controlled | wikipedia_revision | recency_weighted | 2 | 0.500 | 0.500 | 0.500 | NA | NA | NA | 0.000 | 0.000 | 0.000 |
| controlled | wikipedia_revision | minicheck_no_revision | 2 | 0.000 | 0.000 | 0.000 | NA | NA | NA | 0.000 | 0.000 | 0.000 |
| controlled | wikipedia_revision | river_d | 2 | 0.500 | 0.500 | 0.500 | NA | NA | NA | 0.000 | 0.000 | 0.000 |
| controlled | wikipedia_revision | river_l | 10 | 0.400 | 0.400 | 0.400 | NA | NA | NA | 0.000 | 0.000 | 0.000 |

- Authored unit fixtures and synthetic sources are refused for this table.
- abstention_f1 is not applicable for 12 of 42 rows: it needs a predicted revision state, which this system does not emit.
- accuracy is not applicable for 12 of 42 rows: it needs either a resolvable gold answer or a predicted revision state, and this group has neither.
- rfr is not applicable for 28 of 42 rows: it needs examples whose arrival order was reversed, of which this group contains none.
