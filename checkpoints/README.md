# RIVER-L checkpoints

The five RIVER-L checkpoints behind the reported results, one per seed. Each
was trained with `configs/learned/river_l_full.json` on the frozen train split,
with early stopping on the frozen dev split, by
`scripts/release/rerun_human_frozen.sh`.

| File | Seed | Epochs | Best dev loss |
|---|---:|---:|---:|
| `river_l_seed11.pt` | 11 | 23 | 0.7634 |
| `river_l_seed42.pt` | 42 | 21 | 0.7449 |
| `river_l_seed101.pt` | 101 | 36 | 0.6183 |
| `river_l_seed202.pt` | 202 | 30 | 0.6957 |
| `river_l_seed314.pt` | 314 | 33 | 0.6341 |

Run on `data/grounded_human_frozen/river_bench_test.jsonl`, each checkpoint
reproduces the revision state and selected hypothesis of all 248 examples in
the released predictions under `results/human_frozen/`.

Each file stores the model weights, the model and feature configurations and a
training manifest (seed, config hash and train/dev dataset hashes). Verify the
files with `sha256sum -c SHA256SUMS`.

## Usage

```python
from pathlib import Path
from river.learned import load_checkpoint

model, extractor, manifest = load_checkpoint(Path("checkpoints/river_l_seed42.pt"))
```

To evaluate a checkpoint on the frozen test set:

```bash
echo '{"checkpoint": "checkpoints/river_l_seed42.pt", "device": "cpu"}' > river_l_seed42.json
uv run --extra learned python -m experiments.run --baseline river_l \
  --fixture data/grounded_human_frozen/river_bench_test.jsonl \
  --config river_l_seed42.json --seed 42 --no-generation \
  --output river_l_test_seed42.jsonl
```
