#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Replays recorded expert routes against per-layer slot caches of several sizes.

import json
import sys
from collections import OrderedDict


def lru(stream, slots):
    held, hits = OrderedDict(), []
    for experts in stream:
        hit = 0
        for e in experts:
            if e in held:
                held.move_to_end(e)
                hit += 1
            else:
                if len(held) >= slots:
                    for victim in held:
                        if victim not in experts:
                            break
                    del held[victim]
                held[e] = True
        hits.append(hit)
    return hits


def store_policy(stream, slots):
    uses, touched, hits, clock = {}, {}, [], 0
    for experts in stream:
        hit = 0
        for e in experts:
            clock += 1
            if e in uses:
                uses[e] += 1
                touched[e] = clock
                hit += 1
                continue
            if len(uses) >= slots:
                victim = min(
                    (x for x in uses if x not in experts), key=lambda x: (uses[x], touched[x]))
                del uses[victim], touched[victim]
            uses[e], touched[e] = 1, clock
        hits.append(hit)
    return hits


def belady(stream, slots):
    future = {}
    upcoming = [None] * len(stream)
    for t in range(len(stream) - 1, -1, -1):
        upcoming[t] = {e: future.get(e, float("inf")) for e in stream[t]}
        for e in stream[t]:
            future[e] = t
    held, hits = {}, []
    for t, experts in enumerate(stream):
        hit = 0
        for e in experts:
            if e in held:
                hit += 1
            elif len(held) >= slots:
                victim = max((x for x in held if x not in experts), key=lambda x: held[x])
                del held[victim]
            held[e] = upcoming[t][e]
        hits.append(hit)
    return hits


def main():
    data = json.load(open(sys.argv[1]))
    budgets = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else
                                "16,32,64,96,128,192,256,384".split(","))]
    experts, top_k = data["experts"], data["topK"]
    layers = sorted({int(l) for s in data["sessions"] for l in s["decode"]})
    print(f"{experts} experts, top-{top_k}, {len(layers)} sparse layers, "
          f"{len(data['sessions'])} sessions")

    streams, decode_mask = {l: [] for l in layers}, {l: [] for l in layers}
    for session in data["sessions"]:
        for l in layers:
            pre = session["prefill"].get(str(l), [])
            dec = session["decode"].get(str(l), [])
            streams[l] += pre + dec
            decode_mask[l] += [False] * len(pre) + [True] * len(dec)

    print(f"{'slots':>6} {'% bank':>7} {'LRU':>7} {'store':>7} {'Belady':>7}   decode hit rate")
    for slots in budgets:
        if slots < top_k:
            continue
        rates = []
        for policy in (lru, store_policy, belady):
            hit = total = 0
            for l in layers:
                h = policy(streams[l], min(slots, experts))
                for got, counted in zip(h, decode_mask[l]):
                    if counted:
                        hit += got
                        total += top_k
            rates.append(hit / total)
        print(f"{slots:>6} {100 * min(slots, experts) / experts:>6.0f}% "
              + " ".join(f"{100 * r:>6.1f}%" for r in rates))


if __name__ == "__main__":
    main()
