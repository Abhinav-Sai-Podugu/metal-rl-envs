"""v6: DQN, the off-policy learner, environment on GPU vs CPU, swept over N.

Each iteration is one batched environment step and one gradient step on a
128-sample batch from the replay buffer, so N sets how much fresh data each
gradient step gets. Two questions, the off-policy versions of v5's and
v2's: does more fresh data per gradient step cut the gradient steps needed,
and does the environment's location matter now that the learner does one
gradient step per environment step instead of sixteen per thirty-two?
Five seeds, median over solved seeds, unsolved seeds counted; the clock
covers training only, as in v2 and v5.
"""

import argparse

import bench_ppo
import dqn
from bench import RESULTS, environment_line


def sweep(args):
    rows = []
    for backend in args.backends:
        for n in args.ns:
            for seed in args.seeds:
                cfg = dqn.Config(n=n, grad_steps=args.grad_steps, time_budget=args.time_budget)
                r = dqn.train(cfg, backend, seed)
                rows.append({"backend": backend, "series": f"{backend} env", "n": n, "seed": seed, **vars(r)})
                status = f"solved in {r.train_seconds:6.1f}s" if r.solved else f"NOT solved, {r.train_seconds:6.1f}s"
                print(f"{backend:5s} N={n:>6d} seed={seed}  {status}  {r.grad_steps:>7,} grad steps  "
                      f"{r.env_steps:>13,} env steps  score {r.score:5.1f}", flush=True)
    return rows


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ns", type=int, nargs="+", default=[1, 16, 256, 4096, 65536])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--backends", nargs="+", default=["mlx", "numpy"], choices=list(dqn.ENVS))
    p.add_argument("--grad-steps", type=int, default=1, help="gradient steps per iteration")
    p.add_argument("--time-budget", type=float, default=120.0)
    p.add_argument("--out", default="dqn", help="results file stem")
    return p.parse_args()


def main():
    args = parse_args()
    RESULTS.mkdir(exist_ok=True)
    print(environment_line(), flush=True)
    rows = sweep(args)
    series = [f"{b} env" for b in args.backends]
    steps = ("grad_steps", "gradient steps to solve")
    bench_ppo.write_csv(rows, RESULTS / f"{args.out}.csv")
    bench_ppo.plot(rows, args.ns, "series", series, RESULTS / f"{args.out}.png",
                   "DQN on batched CartPole, Q-network on the GPU, environment on GPU vs CPU. Apple M3 Pro", steps=steps)
    print()
    print(bench_ppo.markdown_table(rows, args.ns, "series", series, args.seeds, ("CPU s / GPU s", "numpy env", "mlx env"), steps=steps))
    print()
    cfg = dqn.Config()
    print(f"grad_steps/iter={args.grad_steps} batch={cfg.batch} buffer={cfg.buffer} lr={cfg.lr} eps_decay={cfg.eps_decay} "
          f"target_every={cfg.target_every} learning_starts={cfg.learning_starts} time_budget={args.time_budget}s seeds={args.seeds}")


if __name__ == "__main__":
    main()
