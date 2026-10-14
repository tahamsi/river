import json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

MODEL = sys.argv[1]
URL = "http://127.0.0.1:11434/api/generate"

def call(i):
    body = json.dumps({
        "model": MODEL,
        "prompt": f"Write a short paragraph about knowledge graph entity {i}.",
        "stream": False, "keep_alive": "30m", "options": {"num_predict": 128},
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as resp:
        d = json.loads(resp.read())
    return d.get("eval_count") or 0

call(0)  # warm
for workers in (1, 4, 8, 16):
    start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        tokens = sum(pool.map(call, range(workers)))
    wall = time.time() - start
    print(f"{MODEL} {workers:2d} concurrent -> {wall:6.1f}s wall, {tokens:5d} tok, "
          f"{tokens/wall:6.1f} tok/s aggregate", flush=True)
