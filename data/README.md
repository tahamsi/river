# Data

| Path | Contents |
|---|---|
| `grounded_human_frozen/` | **RIVER-Bench**, the released corpus: 1,502 human-adjudicated examples (751 pairs), split 1,000 / 254 / 248. Documented in [`docs/benchmark.md`](../docs/benchmark.md) |
| `pat_redaction_manifest.json` | What `scripts/benchmark/rehydrate_pat.py` restores, with the SHA-256 of every original file |
| `considerations/ANNOTATION_GUIDE.md` | Evidence and task-eligibility rules for annotation |

## PAT-Questions text

PAT-Questions publishes no licence, so this release does not redistribute its
text. Each PAT question and question template is replaced by a placeholder such
as `@@PAT:3f2a9c1b7d4e0a58:r@@`. Answers, Wikidata identifiers and all RIVER
annotations are kept. Restore the text once after cloning:

```bash
uv run python scripts/benchmark/rehydrate_pat.py
```

The script downloads the PAT-Questions March 2024 snapshot at the pinned
upstream commit, checks its SHA-256, fills in every placeholder, and checks each
restored file against the SHA-256 of the original. The restored files are
byte-identical to the ones the reported results were computed on. It rewrites
the corpus here and the two archives under `results/human_frozen/`. Use
`--pat-dir` with a local checkout of PAT-Questions at that commit to run
offline, and `--check` to verify without writing.

## Not in this release

The corpora `grounded_human_frozen/` was built from (`grounded/`,
`grounded_nli/`, `grounded_expired/`, `grounded_expired_nli/`), the earlier
label approvals and the source-record archive are not distributed. They repeat
the same examples in earlier forms and contain PAT-Questions text throughout.
Tests that check them skip when they are absent.

## Attribution

Every example preserves its `source_dataset`, `source_id`, source-native record
and hashes. The upstream datasets and their licences are listed in
[`docs/benchmark.md`](../docs/benchmark.md#composition); downstream use should
retain those notices.
