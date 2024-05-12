import argparse
import numpy as np
import os
import pickle
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
    asst_scores_badges = np.load(os.path.join(data_dir, "asst_scores_badges.npy"))
    asst_scores_user_embs = np.load(os.path.join(data_dir, "asst_scores_user_embs.npy"))

    covs = pickle.load(open(os.path.join(data_dir, "covs.pkl"), 'rb'))
    loads = pickle.load(open(os.path.join(data_dir, "loads.pkl"), 'rb'))
    user_rep_scores = np.load(os.path.join(data_dir, "user_rep_scores.npy"))
    kp_matching_scores = np.load(os.path.join(data_dir, "kp_matching_scores.npy"))
    sim_scores = np.load(os.path.join(data_dir, "sim_scores.npy"))

    # Select a subset of users and questions to work with (maybe like .5 or .6
    # fraction of each?)
    rng = np.random.default_rng(seed=seed)
    num_e, num_q = asst_scores.shape
    frac = .6
    chosen_experts = rng.choice(range(num_e), size=int(frac*num_e), replace=False)
    chosen_queries = rng.choice(range(num_q), size=int(frac*num_q), replace=False)

    alloc_fname = os.path.join(data_dir, "alloc_%d.npy" % seed)

    rand_fname = os.path.join(data_dir, "alloc_rand_%d_%d.npy" % (1, seed))

    if not os.path.exists(rand_fname):
        np.save(os.path.join(data_dir, "chosen_experts_%d.npy" % seed), chosen_experts)
        np.save(os.path.join(data_dir, "chosen_queries_%d.npy" % seed), chosen_queries)

        asst_scores = asst_scores[chosen_experts, :]
        asst_scores = asst_scores[:, chosen_queries]
        asst_scores_badges = asst_scores_badges[chosen_experts, :]
        asst_scores_badges = asst_scores_badges[:, chosen_queries]
        asst_scores_user_embs = asst_scores_user_embs[chosen_experts, :]
        asst_scores_user_embs = asst_scores_user_embs[:, chosen_queries]
        covs = covs[chosen_queries]
        loads = loads[chosen_experts]
        user_rep_scores = user_rep_scores[chosen_experts, :]
        user_rep_scores = user_rep_scores[:, chosen_queries]
        kp_matching_scores = kp_matching_scores[chosen_experts, :]
        kp_matching_scores = kp_matching_scores[:, chosen_queries]
        sim_scores = sim_scores[chosen_experts, :]
        sim_scores = sim_scores[:, chosen_queries]

        # Now just compute all the assignments, and save them out.
        print("Data loaded. Starting allocations", flush=True)
        fname = os.path.join(data_dir, "alloc_%d.npy" % seed)
        if not os.path.exists(fname):
            est_usw, alloc = solve_usw_gurobi(asst_scores, covs, loads)
            print("Finished with pred asst, est_usw is ", est_usw, flush=True)
            np.save(fname, alloc)
        else:
            print("Skipping pred")

        fname = os.path.join(data_dir, "alloc_badges_%d.npy" % seed)
        if not os.path.exists(fname):
            est_usw, alloc = solve_usw_gurobi(asst_scores_badges, covs, loads)
            print("Finished with pred asst with badges, est_usw is ", est_usw, flush=True)
            np.save(fname, alloc)
        else:
            print("Skipping badges")

        fname = os.path.join(data_dir, "alloc_user_embs_%d.npy" % seed)
        if not os.path.exists(fname):
            est_usw, alloc = solve_usw_gurobi(asst_scores_user_embs, covs, loads)
            print("Finished with pred asst with user embs, est_usw is ", est_usw, flush=True)
            np.save(fname, alloc)
        else:
            print("Skipping user embs")

        # for lam in np.arange(0, 1.01, .1):
        for lam in range(11):
            print("Starting on lambda=", lam, flush=True)
            fname = os.path.join(data_dir, "alloc_non_pred_%d_%d.npy" % (lam, seed))
            if not os.path.exists(fname):
                lambda_val = lam*.1
                non_pred_scores = lambda_val * user_rep_scores / np.max(user_rep_scores)
                # non_pred_scores += (1 - lambda_val) * kp_matching_scores / np.max(kp_matching_scores)
                non_pred_scores += (1 - lambda_val) * sim_scores / np.max(sim_scores)

                _, alloc_non_pred = solve_usw_gurobi(non_pred_scores, covs, loads)
                np.save(fname, alloc_non_pred)
            else:
                print("Skipping nonpred %d" % lam)

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
