# FastGraph: start here

## Human: copy these commands

Reserve a **whole H100 and 64 GiB CPU RAM**, load Conda and CUDA 12.4/compiler
modules, and choose a NEW private work directory with 100 GiB free.
Linux, git, Python 3.10+ and `nvcc` are required.

```bash
git clone --branch codex/gpu-operator-handoff https://github.com/AgarwalAarush/FastGraphCompute.git fgc-gpu-campaign
cd fgc-gpu-campaign
bash revision_campaign/start.sh --work "$HOME/fastgraph-work" --gpu H100 --full-gpu --execute
```

**The startup command downloads pinned sources, installs four environments,
builds the code, runs CPU/GPU checks and the synthetic pilot, and exports a ZIP.**
Use `--gpu A100` for A100, or `--site-env /absolute/path/modules.sh` for site setup.

**First-time installs/builds need a longer allocation.** They are additional to
the 25-minute pilot budget. Use a 30-minute debug slot after setup is ready.
Repeat the identical command to resume; keep the code/work directory unchanged.
Exit 75 means partial progress; inspect any other error. Logs stay under the
work directory; the script never submits more GPU allocations.

**Send back the ZIP printed at the end**, including partial results.
The synthetic pilot needs no detector data or model checkpoint.

## Production

Follow the [production instructions](OPERATOR_DETAILS.md#production-all-configured-results)
to supply real inputs and create a separate reviewed plan. Then, in a four-hour allocation:

```bash
FGC_CORE_PYTHON="$HOME/fastgraph-work/envs/core/bin/python" bash revision_campaign/run_all.sh --plan /absolute/path/production.plan.json --minutes 235 --execute
```

Use your actual interpreter/plan paths. Repeat this command across allocations;
it resumes completed work and exports a ZIP each time. Budget **32–66 A100
GPU-hours estimated; reserve about 70**, plus setup. H100 time is unknown.

**GravNet still needs the real model/checkpoint, adapter and confirmed event data.**
The available model stages measure inference and training steps; they do not
train to convergence. Synthetic benchmarks can run now.

**Agent:** read [AGENT_HANDOFF.md](AGENT_HANDOFF.md).
[Full guide](OPERATOR_DETAILS.md) · [Manual setup](SETUP.md) ·
[Return results/graphs](RETURN_RESULTS.md) · [Validation](VALIDATION.md).
