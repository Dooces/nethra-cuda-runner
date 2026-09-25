"""Cost and depth of the contract core as experience grows (g_min 0, admission seed 14, one push per
present Nethra per interval, exposure only).  Reports Nethra count, construction rate, ms/interval,
and recursive depth (relation-of-relation) of what is actually refound in closure."""
import sys, time, random, _path, nethra_presence as core
def depth(n, memo):
    if n in memo: return memo[n]
    memo[n] = 0
    v = 0 if not n.routes else 1 + max((depth(x, memo) for r in n.routes for x in r), default=0)
    memo[n] = v; return v
def run(name, stream, K, report_every):
    f = core.NethraField(g_min=0.0, admission_seed=14.0); L = [f.new() for _ in range(K)]
    t0 = time.process_time(); last_t = t0; last_n = K; rows = []
    for i, present in enumerate(stream, 1):
        for x in present: L[x].push(1.0)
        f.step(1.0)
        if i % report_every == 0:
            now = time.process_time(); memo = {}
            live = [n for n in f.previous_closure if n.routes]
            rows.append(f"{i}: Nethra {len(f.nethra)} (+{len(f.nethra)-last_n}) {1000*(now-last_t)/report_every:.1f}ms/int "
                        f"refound-depth {max((depth(n, memo) for n in live), default=0)}")
            last_t = now; last_n = len(f.nethra)
    print(f"[{name}]  " + " | ".join(rows), flush=True)
rng = random.Random(1)
for K in (4, 8, 16):
    run(f"random K={K}", [[rng.randrange(K)] for _ in range(1500)], K, 500)
run("cycle 8", [[i % 8] for i in range(3000)], 8, 1000)
# hierarchical: words of 3 letters, sentences of 3 words, a persistent 'topic' Nethra per sentence
words = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 4, 8]]
sent = [[0, 1, 2], [2, 3, 0], [1, 3, 2]]
stream = []
for _ in range(120):
    si = rng.randrange(3)
    for w in sent[si]:
        for letter in words[w]:
            stream.append([letter, 9 + w, 13 + si])          # letter + its word + the sentence topic, co-present
run("letters+words+topics (co-present)", stream, 16, 1080)
