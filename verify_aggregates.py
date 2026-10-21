#!/usr/bin/env python3
"""Recompute figs/curve_data.json from the raw counter CSVs and FAIL LOUDLY
on any mismatch.

This is the missing link of the chain: the figure scripts read
curve_data.json; this script proves curve_data.json against data/. It
implements exactly the formula stated in the README:

    E_net = [E(t1) - E(t0)] - (dI / 15.0) * t_active

summed over the GPU pair of the route (NVLink: GPUs 0,1; SYS: GPUs 0,2),
with dI the energy of the idle window preceding each condition and 15.0 s
the scripted idle duration (the paper's Section IV discusses the
timestamped value). Aggregates are the mean over the pooled replicate
windows of the three sweep runs; *_sd fields are the sample SD.

Zero dependencies beyond the Python 3 standard library. Exit code 0 means
every published aggregate reproduces from the raw counters; any divergence
beyond tolerance aborts with a nonzero exit code.
"""
import csv
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
POOL = ("curve_136032", "run_136034", "run_136281")
IDLE_S = 15.0       # scripted idle duration (see README / paper Section IV)

# curve_data.json stores J/J_sd rounded to 4 decimals and P/P_sd/G/G_sd to 2.
# The honest tolerance is therefore half a unit of the last published digit;
# anything beyond that is a real divergence, not rounding.
TOL = {"J": 5e-5, "J_sd": 5e-5,
       "P": 5e-3, "P_sd": 5e-3, "G": 5e-3, "G_sd": 5e-3}


def fail(msg):
    sys.stderr.write("VERIFY FAILED: %s\n" % msg)
    sys.exit(1)


def load_energy(path):
    """snapshot -> {gpu_index: total_energy_mJ}"""
    snaps = {}
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) < 9 or row[1].startswith("#"):
                continue
            snap, _, gpu, _, _, _, name, ret, val = row[:9]
            if ret == "SUCCESS" and name == "total_energy_mJ":
                snaps.setdefault(snap, {})[int(gpu)] = int(val)
    return snaps


def window_dE(snaps, tag, pair):
    t0, t1 = snaps.get(tag + "_t0"), snaps.get(tag + "_t1")
    if t0 is None or t1 is None:
        return None
    for g in pair:
        if g not in t0 or g not in t1:
            fail("GPU %d missing in snapshots of %s" % (g, tag))
    return sum(t1[g] - t0[g] for g in pair) / 1000.0   # mJ -> J


def main():
    buckets = {}   # (route, pattern, mib) -> list of (J_per_GB, P_net, GBs)
    for job in POOL:
        jdir = os.path.join(HERE, "data", job)
        snaps = load_energy(os.path.join(jdir, "energy.csv"))
        with open(os.path.join(jdir, "bench.csv")) as f:
            for row in csv.reader(f):
                if len(row) != 12:
                    fail("unexpected bench.csv format in %s (%d cols)"
                         % (job, len(row)))
                tag, src, dst = row[0], int(row[1]), int(row[2])
                mib, pattern = int(row[3]), row[4]
                nbytes, secs = int(row[9]), float(row[10])
                gbs = float(row[11])
                route = tag.split("_")[0]
                pair = (src, dst)
                dE = window_dE(snaps, tag, pair)
                dI = window_dE(snaps, "idlepre_" + tag, pair)
                if dE is None or dI is None:
                    fail("missing energy window for %s in %s" % (tag, job))
                e_net = dE - (dI / IDLE_S) * secs
                if e_net <= 0:
                    fail("non-positive net energy in %s/%s" % (job, tag))
                j_per_gb = e_net / (nbytes / 1e9)
                buckets.setdefault((route, pattern, mib), []).append(
                    (j_per_gb, e_net / secs, gbs))

    published = json.load(open(os.path.join(HERE, "figs", "curve_data.json")))
    if len(published) != len(buckets):
        fail("key count differs: published %d, recomputed %d"
             % (len(published), len(buckets)))

    checked = 0
    for (route, pattern, mib), rows in sorted(buckets.items()):
        key = "%s|%s|%d" % (route, pattern, mib)
        if key not in published:
            fail("recomputed key %s absent from curve_data.json" % key)
        pub = published[key]
        js, ps, gs = zip(*rows)
        got = {
            "n": len(rows),
            "J": statistics.mean(js), "J_sd": statistics.stdev(js),
            "P": statistics.mean(ps), "P_sd": statistics.stdev(ps),
            "G": statistics.mean(gs), "G_sd": statistics.stdev(gs),
        }
        if got["n"] != pub["n"]:
            fail("%s: n=%d published, %d recomputed" % (key, pub["n"], got["n"]))
        for field in ("J", "J_sd", "P", "P_sd", "G", "G_sd"):
            tol = TOL[field] + 1e-9
            if abs(got[field] - pub[field]) > tol:
                fail("%s: %s published %.6g, recomputed %.6g (tol %.2g)"
                     % (key, field, pub[field], got[field], tol))
            checked += 1

    print("OK: %d aggregates over %d keys reproduce from the raw counters "
          "(tolerance: half a unit of the last published digit)."
          % (checked, len(buckets)))


if __name__ == "__main__":
    main()
