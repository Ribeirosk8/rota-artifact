# ROTA: measurement code, raw energy-counter data, and figure scripts

Artifact of the paper **"ROTA: Measuring What Throughput Hides in the Energy
Cost of Multi-GPU Communication Routes"**, accepted as a short paper at the
13th IEEE/ACM International Conference on Big Data Computing, Applications
and Technologies (BDCAT 2026), Florianopolis, Brazil, December 2026.

Authors: Amauri Ribeiro, Jie Li, Alan Sill and Yong Chen (Texas Tech
University). Measurements were taken on the H100 nodes of the REPACSS cluster
at Texas Tech University.

## Contents

- `code/`: the measurement programs and batch scripts used in the paper:
  - `route_bench.cu`: CUDA benchmark that moves data over a selected
    GPU-to-GPU route (`cudaMemcpyPeerAsync`) while reading the NVML
    accumulated energy counter (`nvmlDeviceGetTotalEnergyConsumption`)
    and per-window SM clock, temperature, and NVLink byte counters.
  - `curve.slurm`: the message-size sweep (both routes, const and random
    fills, three replicates per condition, randomized order, recorded seed).
  - `cachetest.slurm`: the working-set experiment (Table I).
  - `nccl.slurm`: the NCCL AllReduce experiment (Section IV).
- `data/`: raw counter data for every run reported in the paper:
  - `curve_136032/`, `run_136034/`, `run_136281/`: the three independent
    sweep runs (the paper's pool; 288 message-size windows).
  - `cache_137011/`: the working-set run behind Table I.
  - `nccl_140031/`: the NCCL AllReduce run behind Section IV.
  - `validate_135951/`, `sys6_136033/`, `curve_138135/`: the remaining runs
    of the measurement campaign (a 12-window mixed validation run, a
    6-replicate SYS run at 256 MiB, and an additional 36-window sweep
    replicate at 16/64/256 MiB). Together with the pool and the working-set
    run these are the 375 windows / 3.10 PB stated in Section II; the NCCL
    run is counted separately in Section IV.
  - `curve_143515/` through `curve_143520/`: the six-node replication behind
    the fleet numbers in the Threats section (random data costing 13.4% to
    14.5% more energy per byte than constant on every node). One node each,
    96 windows per node: both routes, const and random fills, four
    repetitions per condition at the six principal sizes, all under the
    uniform 1785 MHz clock regime discussed there. These runs are separate
    from the 375-window campaign above.
  - Each directory holds `bench.csv` (per-window bytes, elapsed time,
    throughput) and `energy.csv` (NVML snapshots at window start/end,
    including the idle window preceding each condition).
- `figs/`: the exact scripts that generate Figures 1 to 3 from
  `curve_data.json`, plus `curve_data.json` itself (per-condition mean and
  SD over replicate windows).

## Reproducing the numbers and figures

```
./reproduce.sh
```

This first runs `verify_aggregates.py`, which recomputes every aggregate in
`figs/curve_data.json` from the raw counter CSVs in `data/` (formula below)
and aborts on any divergence beyond half a unit of the last published digit;
it then regenerates Figures 1 to 3. Analysis-only, offline, deterministic.
Requires Python 3 with matplotlib and numpy only (the verifier itself uses
only the standard library).

Note on re-measurement: the scripts in `code/` can re-run the campaign on a
Slurm cluster with H100 GPUs, but the measured operating point (the
per-device SM clocks reported in Section II) cannot be pinned in user space
on the measured system (every NVML clock-control interface returns
`NO_PERMISSION`, see Section IV), so re-measured absolute values will
reflect the then-current clock regime.

## Recomputing the aggregates from raw counters

`curve_data.json` aggregates the raw CSVs with, per window,

```
E_net = [E(t1) - E(t0)] - (dI / 15.0) * t_active
```

where `E` is the accumulated energy counter summed over the GPU pair of the
route (NVLink: GPUs 0,1; SYS: GPUs 0,2), `dI` is the energy of the idle
window preceding the condition, `15.0 s` is the scripted idle duration
(Section IV of the paper discusses the timestamped value, 15.262 s), and
`t_active` is the measured window duration from `bench.csv`. Per-condition
values are the mean over replicate windows; `_sd` fields are the SD over the
same windows. All numbers in the paper derive from these files.

## Provenance notes

- The artifact was first released in anonymized form for double-blind review.
  In the text files of `data/`, hostnames, the cluster name, the institution
  and usernames were replaced with `*-redacted` placeholders at that time. No
  measured value (energy, time, bytes, clocks, temperatures) was altered, and
  the placeholders were kept so that this public copy is byte-identical in
  its data to the reviewed one.
- The per-run lab notes written at the time of each run are not part of this
  public copy; the run descriptions above summarize what each directory holds.

## How to cite

```
@inproceedings{ribeiro2026rota,
  author    = {Ribeiro, Amauri and Li, Jie and Sill, Alan and Chen, Yong},
  title     = {{ROTA}: Measuring What Throughput Hides in the Energy Cost of
               Multi-{GPU} Communication Routes},
  booktitle = {Proc.\ IEEE/ACM BDCAT '26},
  year      = {2026},
  note      = {Short paper},
}
```

## License

Code (`code/`, `figs/*.py`, `verify_aggregates.py`, `reproduce.sh`) is
released under the MIT License. Data (`data/`, `figs/curve_data.json`) and
figures are released under CC BY 4.0. See `LICENSE` and `LICENSE-DATA`.

## Acknowledgment

This research is supported by the National Science Foundation, Office of
Advanced Cyberinfrastructure (Award Number 2404438).
