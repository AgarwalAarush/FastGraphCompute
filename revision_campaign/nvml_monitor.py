"""Separate-process NVML sampler: cannot be starved by the worker's GIL."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time


def main():
    import pynvml as nv
    ready, stop, output, uuid = sys.argv[1:]
    nv.nvmlInit()
    handle = nv.nvmlDeviceGetHandleByUUID(uuid)
    baseline = int(nv.nvmlDeviceGetMemoryInfo(handle).used)
    peak, samples = baseline, 0
    start = time.perf_counter()
    Path(ready).write_text(str(baseline))
    while not Path(stop).exists():
        peak = max(peak, int(nv.nvmlDeviceGetMemoryInfo(handle).used))
        samples += 1
        time.sleep(0.002)
    end = int(nv.nvmlDeviceGetMemoryInfo(handle).used)
    peak = max(peak, end)
    Path(output).write_text(json.dumps({"baseline_bytes": baseline, "peak_used_bytes": peak,
                                       "end_used_bytes": end, "peak_delta_bytes": peak - baseline,
                                       "end_delta_bytes": end - baseline, "samples": samples,
                                       "duration_seconds": time.perf_counter() - start,
                                       "requested_poll_seconds": 0.002,
                                       "method": "separate_process_NVML_device_wide_sampled_peak"}))
    nv.nvmlShutdown()


if __name__ == "__main__":
    main()
