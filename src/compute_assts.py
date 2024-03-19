import argparse
import numpy as np
import os
import sys
sys.path.append("/mnt/nfs/scratch1/jpayan/RAU")
from solve_usw import solve_usw_gurobi


def main(args):
    topic = args.topic
    seed = args.seed
    base_dir = "/mnt/nfs/scratch1/jpayan/predictive_expert_assignment"

    # Open up the folder and load in the asst_scores, covs, loads, kp_matching_scores,
    # user_rep_scores
    data_dir = os.path.join(base_dir, "data", "%s.stackexchange.com" % topic, "npy")
    asst_scores = np.load(os.path.join(data_dir, "asst_scores.npy"))
    covs = np.load(os.path.join(data_dir, "covs.npy"))
    loads = np.load(os.path.join(data_dir, "loads.npy"))
    user_rep_scores = np.load(os.path.join(data_dir, "user_rep_scores.npy"))
    kp_matching_scores = np.load(os.path.join(data_dir, "kp_matching_scores.npy"))

    # Select a subset of users and questions to work with (maybe like .5 or .6
    # fraction of each?)
    rng = np.random.default_rng(seed=seed)
    num_e, num_q = asst_scores.shape
    frac = .6
    chosen_experts = rng.choice(range(num_e), size=int(frac*num_e), replace=False)
    chosen_queries = rng.choice(range(num_q), size=int(frac*num_q), replace=False)

    np.save(os.path.join(data_dir, "chosen_experts_%d.npy" % seed), chosen_experts)
    np.save(os.path.join(data_dir, "chosen_queries_%d.npy" % seed), chosen_queries)

    asst_scores = asst_scores[chosen_experts, :]
    asst_scores = asst_scores[:, chosen_queries]
    covs = covs[chosen_queries]
    loads = loads[chosen_experts]
    user_rep_scores = user_rep_scores[chosen_experts, :]
    user_rep_scores = user_rep_scores[:, chosen_queries]
    kp_matching_scores = kp_matching_scores[chosen_experts, :]
    kp_matching_scores = kp_matching_scores[:, chosen_queries]

    # Now just compute all the assignments, and save them out.
    print("Data loaded. Starting allocations", flush=True)
    est_usw, alloc = solve_usw_gurobi(asst_scores, covs, loads)
    print("Finished with pred asst, est_usw is ", est_usw, flush=True)
    np.save(os.path.join(data_dir, "alloc_%d.npy" % seed), alloc)

    # for lam in np.arange(0, 1.01, .1):
    for lam in range(1):
        print("Starting on lambda=", lam, flush=True)
        lambda_val = lam*.1
        non_pred_scores = lambda_val * user_rep_scores / np.max(user_rep_scores)
        non_pred_scores += (1 - lambda_val) * kp_matching_scores / np.max(kp_matching_scores)
        _, alloc_non_pred = solve_usw_gurobi(non_pred_scores, covs, loads)
        np.save(os.path.join(data_dir, "alloc_non_pred_%d_%d.npy" % (lam, seed)), alloc_non_pred)

    for ridx in range(1):
        print("Starting on ridx=", ridx, flush=True)
        rand_scores = rng.uniform(0, 1, size=alloc.shape)
        _, alloc_rand = solve_usw_gurobi(rand_scores, covs, loads)
        np.save(os.path.join(data_dir, "alloc_rand_%d_%d.npy" % (ridx, seed)), alloc_rand)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", type=str, default="cs")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    main(args)
