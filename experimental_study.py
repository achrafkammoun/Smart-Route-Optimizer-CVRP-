import os, time, random, math, csv
from collections import defaultdict
import statistics, multiprocessing as mp
import numpy as np
import vrplib
from julian_vrplib_genetic import genetic_algorithm


def load_instance_solution(instance_path, solution_path):
    instance = vrplib.read_instance(instance_path)
    sol = vrplib.read_solution(solution_path) if solution_path else None
    return instance, sol


def run_once(instance, seed, run_kwargs):
    random.seed(seed)
    np.random.seed(seed)
    start = time.time()
    result = genetic_algorithm(instance, **run_kwargs)
    runtime = time.time() - start

    # Defensive fallback
    if not isinstance(result, dict):
        result = {"route": None, "cost": float("inf")}

    best_route = result.get("route")
    best_cost = float(result.get("cost", float("inf")))
    n_routes = result.get("n_routes", len(best_route) if best_route else 0)

    return {
        "seed": seed,
        "cost": best_cost,
        "runtime": runtime,
        "n_routes": n_routes,
        "result": result,
    }


def run_experiments(instance_path, solution_path, repeats, sweep, fixed_params, outdir, processes=1):
    os.makedirs(outdir, exist_ok=True)
    instance, solution = load_instance_solution(instance_path, solution_path)
    optimal_cost = solution.get("cost") if solution else None

    # Parameter grid
    grid = []
    keys = list(sweep.keys())
    from itertools import product
    if not keys:
        grid = [fixed_params]
    else:
        for vals in product(*[sweep[k] for k in keys]):
            conf = dict(zip(keys, vals))
            conf.update(fixed_params)
            grid.append(conf)

    runs = []

    def worker_job(args):
        conf, seed = args
        run_kwargs = {k: conf[k] for k in conf if k not in ("_internal",)}
        res = run_once(instance, seed, run_kwargs)
        res.update(conf)
        if optimal_cost is not None:
            res["gap_pct"] = 100.0 * (res["cost"] - optimal_cost) / optimal_cost
        else:
            res["gap_pct"] = None
        return res

    jobs = []
    for conf in grid:
        for _ in range(repeats):
            seed = random.randrange(1 << 30)
            jobs.append((conf.copy(), seed))

    if processes > 1:
        with mp.Pool(processes) as pool:
            for res in pool.imap_unordered(worker_job, jobs):
                runs.append(res)
    else:
        for job in jobs:
            runs.append(worker_job(job))

    if not runs:
        print("⚠️ No runs produced any results.")
        return [], []

    # Save raw runs
    runs_csv = os.path.join(outdir, "runs.csv")
    keys_out = sorted(k for k in runs[0].keys() if k != "result")
    with open(runs_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys_out)
        writer.writeheader()
        for r in runs:
            writer.writerow({k: r.get(k) for k in keys_out})

    # Aggregated summary
    grouped = defaultdict(list)
    for r in runs:
        conf_key = tuple((k, r.get(k)) for k in keys)
        grouped[conf_key].append(r)

    summary_rows = []
    for conf_key, group in grouped.items():
        if isinstance(group, dict):
            group = [group]

        costs = [g["cost"] for g in group if isinstance(g, dict) and "cost" in g]
        runtimes = [g["runtime"] for g in group if isinstance(g, dict) and "runtime" in g]
        gaps = [g["gap_pct"] for g in group if isinstance(g, dict) and g.get("gap_pct") is not None]

        if not costs:
            continue

        mean_cost = statistics.mean(costs)
        std_cost = statistics.stdev(costs) if len(costs) > 1 else 0.0
        median_cost = statistics.median(costs)
        mean_rt = statistics.mean(runtimes)
        ci95 = 1.96 * (std_cost / math.sqrt(len(costs))) if len(costs) > 1 else 0.0
        row = {k: v for k, v in conf_key}
        row.update(
            {
                "n_runs": len(costs),
                "mean_cost": mean_cost,
                "std_cost": std_cost,
                "median_cost": median_cost,
                "ci95_cost": ci95,
                "mean_runtime": mean_rt,
                "mean_gap_pct": statistics.mean(gaps) if gaps else None,
            }
        )
        summary_rows.append(row)

    summary_csv = os.path.join(outdir, "summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)

    return runs, summary_rows
