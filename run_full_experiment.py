import os
import statistics
import math
from experimental_study import run_experiments

EXPERIMENTS = [
    {
        "instance": "M-n200-k16.vrp",
        "solution": "M-n200-k16.sol",
        "repeats": 10,
        "sweep": {},
        "fixed": {"pop_size": 100, "generations": 1000, "mutation_rate": 0.3, "elitism": 4},
        "outdir": "C101"
    }
]

def main():
    import matplotlib.pyplot as plt
    import numpy as np

    for exp in EXPERIMENTS:
        print(f"\n=== Running experiment for {exp['instance']} ===\n")

        # Run experiments one by one and print each run immediately
        runs = []
        for i in range(1, exp["repeats"] + 1):
            run_result, _ = run_experiments(
                exp["instance"], exp["solution"],
                1,  # run one at a time
                exp["sweep"], exp["fixed"], exp["outdir"]
            )
            run = run_result[0]
            runs.append(run)

            gap = run["gap_pct"]
            runtime = run["runtime"]
            print(f"Run {i:2d}: runtime={runtime:.3f}s, gap={gap if gap is not None else 'infeasible'}%")

        # Prepare data for statistics and plots
        runtimes = [r["runtime"] for r in runs]
        gaps_numeric = [r["gap_pct"] for r in runs if r["gap_pct"] is not None]  # numeric gaps only
        gaps_plot = [r["gap_pct"] if r["gap_pct"] is not None else np.nan for r in runs]

        name = os.path.basename(exp["instance"]).replace(".vrp", "")

        def ci95(x):
            return 1.96 * statistics.stdev(x) / math.sqrt(len(x)) if len(x) > 1 else 0.0

        # Print summary stats
        print(f"\n{name} summary stats:")
        print(f"  mean runtime: {statistics.mean(runtimes):.3f}s ± {ci95(runtimes):.3f}")
        print(f"  median runtime: {statistics.median(runtimes):.3f}s")
        if gaps_numeric:
            print(f"  mean gap: {statistics.mean(gaps_numeric):.3f}% ± {ci95(gaps_numeric):.3f}")
            print(f"  median gap: {statistics.median(gaps_numeric):.3f}%")
        else:
            print("  No feasible runs to calculate gap statistics.")

        # Create plots
        plots_dir = os.path.join(exp["outdir"], "plots")
        os.makedirs(plots_dir, exist_ok=True)

        plt.figure(figsize=(8, 6))
        plt.boxplot([g for g in gaps_plot if not math.isnan(g)])
        plt.ylabel("Gap (%)")
        plt.title(f"{name}: Gap distribution")
        plt.savefig(os.path.join(plots_dir, "gap_boxplot_full.png"))
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.boxplot(runtimes)
        plt.ylabel("Runtime (s)")
        plt.title(f"{name}: Runtime distribution")
        plt.savefig(os.path.join(plots_dir, "runtime_boxplot_full.png"))
        plt.close()

    print("\n✅ All experiments done.")

if __name__ == "__main__":
    main()
