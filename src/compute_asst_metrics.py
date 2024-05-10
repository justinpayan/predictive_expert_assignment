import argparse
import cvxpy as cp
import numpy as np
import pickle
import os
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity


def est_dens_ratio(input_vects, reference_vects):
    pca_ref = PCA(n_components=2).fit(reference_vects)
    kde_ref = KernelDensity().fit(pca_ref.transform(reference_vects))

    pca = PCA(n_components=2).fit_transform(input_vects)
    kde = KernelDensity().fit(pca)
    log_density = kde.score_samples(pca)
    probs = np.exp(log_density)
    log_density_input_by_ref = kde_ref.score_samples(pca_ref.transform(input_vects))
    probs_input_by_ref = np.exp(log_density_input_by_ref)
    return probs / probs_input_by_ref


def get_extreme_case(fn, preds_score, dens_ratio, emp_loss, n_samples, delta,
                   num_pap, cov_constr, gamma):
    v = cp.Variable(preds_score.shape)

    eta = 1 / n_samples
    eta += np.sum(dens_ratio ** (-2)) / (cov_constr * num_pap) ** 2
    eta *= np.log(1 / delta) / 2
    eta = gamma * np.sqrt(eta)

    rhs = preds_score.shape[0] * (emp_loss + eta)
    lhs = cp.multiply(v, np.log(preds_score)) + cp.multiply(1 - v, np.log(1 - preds_score))
    lhs *= -1
    lhs = cp.multiply(lhs, dens_ratio ** (-1))
    adv_prob = cp.Problem(fn(cp.sum(v)),
                          [cp.sum(lhs) <= rhs, v >= np.zeros(v.shape), v <= np.ones(v.shape)])
    adv_prob.solve(verbose=True, solver='GUROBI')

    return v.value


def get_worst_case(preds_score, dens_ratio, emp_loss, n_samples, delta,
                   num_pap, cov_constr, gamma=1.3):
    return get_extreme_case(cp.Minimize, preds_score, dens_ratio, emp_loss, n_samples, delta,
                     num_pap, cov_constr, gamma)


def get_best_case(preds_score, dens_ratio, emp_loss, n_samples, delta,
                   num_pap, cov_constr, gamma=1.3):
    return get_extreme_case(cp.Maximize, preds_score, dens_ratio, emp_loss, n_samples, delta,
                     num_pap, cov_constr, gamma)


def main(args):
    topic = args.topic
    seed = args.seed
    base_dir = "/mnt/nfs/scratch1/jpayan/predictive_expert_assignment"

    # Open up the folder and load in the asst_scores, covs, loads, kp_matching_scores,
    # user_rep_scores
    data_dir = os.path.join(base_dir, "data", "%s.stackexchange.com" % topic, "npy")
    asst_scores = np.load(os.path.join(data_dir, "asst_scores.npy"))
    covs = pickle.load(open(os.path.join(data_dir, "covs.pkl"), 'rb'))
    loads = pickle.load(open(os.path.join(data_dir, "loads.pkl"), 'rb'))
    user_rep_scores = np.load(os.path.join(data_dir, "user_rep_scores.npy"))
    kp_matching_scores = np.load(os.path.join(data_dir, "kp_matching_scores.npy"))
    past_usefulness = np.load(os.path.join(data_dir, "past_usefulness.npy"))
    past_relevance = np.load(os.path.join(data_dir, "past_relevance.npy"))
    past_informativeness = np.load(os.path.join(data_dir, "past_informativeness.npy"))
    estimated_topical_sim = np.load(os.path.join(data_dir, "estimated_topical_sim.npy"))
    fifth_percentile = np.load(os.path.join(data_dir, "fifth_percentile.npy"))
    median = np.load(os.path.join(data_dir, "median.npy"))
    realScores = np.load(os.path.join(data_dir, "realScores.npy"))
    hasScore = np.load(os.path.join(data_dir, "hasScore.npy"))
    pair_to_feats = pickle.load(open(os.path.join(data_dir, "pair_to_feats.pkl"), 'rb'))
    scaled_test = np.load(os.path.join(data_dir, "scaled_test.npy"))
    xe_test = pickle.load(open(os.path.join(data_dir, "xe_test.pkl"), 'rb'))

    # Select a subset of users and questions to work with
    chosen_experts = np.load(os.path.join(data_dir, "chosen_experts_%d.npy" % seed))
    chosen_queries = np.load(os.path.join(data_dir, "chosen_queries_%d.npy" % seed))

    asst_scores = asst_scores[chosen_experts, :]
    asst_scores = asst_scores[:, chosen_queries]
    covs = covs[chosen_queries]
    loads = loads[chosen_experts]
    user_rep_scores = user_rep_scores[chosen_experts, :]
    user_rep_scores = user_rep_scores[:, chosen_queries]
    kp_matching_scores = kp_matching_scores[chosen_experts, :]
    kp_matching_scores = kp_matching_scores[:, chosen_queries]
    past_relevance = past_relevance[chosen_experts, :]
    past_relevance = past_relevance[:, chosen_queries]
    past_informativeness = past_informativeness[chosen_experts, :]
    past_informativeness = past_informativeness[:, chosen_queries]
    past_usefulness = past_usefulness[chosen_experts, :]
    past_usefulness = past_usefulness[:, chosen_queries]
    estimated_topical_sim = estimated_topical_sim[chosen_experts, :]
    estimated_topical_sim = estimated_topical_sim[:, chosen_queries]
    fifth_percentile = fifth_percentile[chosen_experts, :]
    fifth_percentile = fifth_percentile[:, chosen_queries]
    median = median[chosen_experts, :]
    median = median[:, chosen_queries]
    realScores = realScores[chosen_experts, :]
    realScores = realScores[:, chosen_queries]
    hasScore = hasScore[chosen_experts, :]
    hasScore = hasScore[:, chosen_queries]

    # Load up the allocations
    print("Data loaded. Loading allocations", flush=True)
    pred_alloc = np.load(os.path.join(data_dir, "alloc_%d.npy" % seed))
    pred_alloc_user_embs = np.load(os.path.join(data_dir, "alloc_user_embs_%d.npy" % seed))
    pred_alloc_badges = np.load(os.path.join(data_dir, "alloc_badges_%d.npy" % seed))

    non_pred_allocs = []

    for lam in range(11):
        print("Loading lambda=", lam, flush=True)
        non_pred_allocs.append(np.load(
            os.path.join(data_dir, "alloc_non_pred_%d_%d.npy" % (lam, seed))))

    rand_allocs = []
    for ridx in range(1):
        print("Loading ridx=", ridx, flush=True)
        rand_allocs.append(
            np.load(os.path.join(data_dir, "alloc_rand_%d_%d.npy" % (ridx, seed))))

    metric_to_allocation_scores = {}

    total_assts = pred_alloc.shape[1]*covs[0]

    # Compute remaining metrics
    metric_names = ['Keyword', 'User Rep.', 'Usefulness', 'Relevance', 'Informativeness', 'Similarity',
                    '$5^{th}$ \\%-ile', 'Median', 'Pred.']
    metric_matrices = [kp_matching_scores, user_rep_scores, past_usefulness, past_relevance,
                       past_informativeness, estimated_topical_sim, fifth_percentile, median, asst_scores]

    for metric_name, metric_matrix in zip(metric_names, metric_matrices):
        metric_to_allocation_scores[metric_name] = {}

        for idx, alloc_non_pred in enumerate(non_pred_allocs):
            metric_to_allocation_scores[metric_name][idx] = np.sum(alloc_non_pred * metric_matrix) / total_assts
        metric_to_allocation_scores[metric_name]['pred'] = np.sum(pred_alloc * metric_matrix) / total_assts
        metric_to_allocation_scores[metric_name]['pred_user_embs'] = np.sum(pred_alloc_user_embs * metric_matrix) / total_assts
        metric_to_allocation_scores[metric_name]['pred_badges'] = np.sum(pred_alloc_badges * metric_matrix) / total_assts

        for idx, alloc_rand in enumerate(rand_allocs):
            metric_to_allocation_scores[metric_name]["rand" + str(idx)] = np.sum(alloc_rand * metric_matrix) / total_assts

    metric_name = 'Avg. Vote'
    metric_to_allocation_scores[metric_name] = {}
    for idx, alloc_non_pred in enumerate(non_pred_allocs):
        metric_to_allocation_scores[metric_name][idx] = np.sum(alloc_non_pred * realScores) / np.sum(
            alloc_non_pred * hasScore)
    for idx, alloc_rand in enumerate(rand_allocs):
        metric_to_allocation_scores[metric_name]["rand" + str(idx)] = np.sum(alloc_rand * realScores) / np.sum(
            alloc_rand * hasScore)
    metric_to_allocation_scores[metric_name]['pred'] = np.sum(pred_alloc * realScores) / np.sum(pred_alloc * hasScore)
    metric_to_allocation_scores[metric_name]['pred_user_embs'] = np.sum(pred_alloc_user_embs * realScores) / np.sum(pred_alloc_user_embs * hasScore)
    metric_to_allocation_scores[metric_name]['pred_badges'] = np.sum(pred_alloc_badges * realScores) / np.sum(pred_alloc_badges * hasScore)

    metric_name = 'Hits'
    metric_to_allocation_scores[metric_name] = {}
    for idx, alloc_non_pred in enumerate(non_pred_allocs):
        metric_to_allocation_scores[metric_name][idx] = np.sum(alloc_non_pred * hasScore)
    for idx, alloc_rand in enumerate(rand_allocs):
        metric_to_allocation_scores[metric_name]["rand" + str(idx)] = np.sum(alloc_rand * hasScore)
    metric_to_allocation_scores[metric_name]['pred'] = np.sum(pred_alloc * hasScore)
    metric_to_allocation_scores[metric_name]['pred_user_embs'] = np.sum(pred_alloc_user_embs * hasScore)
    metric_to_allocation_scores[metric_name]['pred_badges'] = np.sum(pred_alloc_badges * hasScore)

    print("Done computing standard metrics, moving to lower and upper bounds", flush=True)

    # Compute the worst and best case scores
    metric_to_allocation_scores['best_usw'] = {}
    metric_to_allocation_scores['worst_usw'] = {}

    all_allocs = [pred_alloc, pred_alloc_user_embs, pred_alloc_badges] + non_pred_allocs + rand_allocs
    all_names = ['pred', 'pred_user_embs', 'pred_badges'] + list(range(11)) + ["rand" + str(i) for i in range(1)]

    delta = .1
    n_samples = len(scaled_test)

    prob_up = (asst_scores + 5)/6

    for alloc, alloc_name in zip(all_allocs, all_names):
        collection_alloc = []

        for e, q in zip(np.where(alloc)[0], np.where(alloc)[1]):
            collection_alloc.append(pair_to_feats[(e, q)])
        l = np.array(collection_alloc)
        normalizer1 = np.max(l, axis=0)
        l /= normalizer1
        normalizer = np.max(np.linalg.norm(l, axis=1))
        l /= normalizer

        dr_alloc = est_dens_ratio(l, scaled_test)

        which_entries = np.where(alloc)

        v_worst = get_worst_case(prob_up[which_entries], dr_alloc, xe_test,
                                 n_samples, delta, pred_alloc.shape[1], covs[0])
        v_best = get_best_case(prob_up[which_entries], dr_alloc, xe_test,
                               n_samples, delta, pred_alloc.shape[1], covs[0])

        worst_usw = np.sum(6 * v_worst - 5) / total_assts
        best_usw = np.sum(6 * v_best - 5) / total_assts

        metric_to_allocation_scores['best_usw'][alloc_name] = best_usw
        metric_to_allocation_scores['worst_usw'][alloc_name] = worst_usw

        pickle.dump(metric_to_allocation_scores,
                    open(os.path.join(data_dir, "metric_to_allocation_scores_%d.pkl" % seed), 'wb'))
        print("Worst/best for %s is %.2f, %.2f" % (alloc_name, worst_usw, best_usw), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", type=str, default="cs")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    main(args)
