import argparse
import cvxpy as cp
import gurobipy as gp
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


def norm_zero_one(X_all):
    assert X_all.shape[1] <= 3
    for dim in range(X_all.shape[1]):
        Xi = X_all[:, dim]
        mask = ~np.isnan(Xi)
        a = np.min(Xi[mask])
        b = np.max(Xi[mask])
        X_all[:, dim] = (Xi - a) / (b - a)


def compute_ymin_ymax_monotonic(alloc, realScores, hasScore, userRep, estSimScores, kpScores):
    observed = np.where(hasScore > .5)
    obs_idx = []
    for i, j in zip(observed[0], observed[1]):
        obs_idx.append(i * alloc.shape[1] + j)
    alloc_but_not_obs = np.where((alloc > .5) & (hasScore < .5))
    vio_idx = []
    for i, j in zip(alloc_but_not_obs[0], alloc_but_not_obs[1]):
        vio_idx.append(i * alloc.shape[1] + j)

    X = np.column_stack((userRep.flatten(), estSimScores.flatten(), kpScores.flatten()))
    norm_zero_one(X)

    X_obs = X[obs_idx, :]
    X_vio = X[vio_idx, :]

    def cdom(X1, X2):
        weak = np.all(X1[:, np.newaxis, :] >= X2[np.newaxis, :, :], axis=-1)
        assert weak.shape == (X1.shape[0], X2.shape[0])
        strong = np.any(X1[:, np.newaxis, :] > X2[np.newaxis, :, :], axis=-1)
        assert strong.shape == (X1.shape[0], X2.shape[0])
        dom = (weak & strong)
        return dom  # dom_ij == 1 <-> i dom j

    X_obs_vio = np.concatenate((X_obs, X_vio))
    dom = cdom(X_obs_vio, X_obs_vio)

    m = gp.Model()

    m.setParam('Method', 1)
    N_obs = int(np.sum(hasScore))
    N_vio = np.sum((hasScore < .5) & (alloc > .5))
    obj_T = np.zeros(X_obs_vio.shape[0])

    obj_T[N_obs:] = 1
    y_min = -5
    y_max = 1
    T = m.addMVar(N_obs + N_vio, lb=y_min, ub=y_max, obj=obj_T)

    lam = 1e9
    delta_obs = m.addMVar(N_obs, lb=0, ub=y_max - y_min, obj=lam)
    m.modelSense = gp.GRB.MINIMIZE

    Y_obs = realScores[hasScore > .5]
    m.addConstr(T[:N_obs] - Y_obs <= delta_obs)
    m.addConstr(Y_obs - T[:N_obs] <= delta_obs)

    mask = np.zeros((N_obs + N_vio, N_obs + N_vio))
    I = np.eye(N_obs + N_vio)
    LB = np.full(dom.shape, y_min - y_max)  # no bound if not dom, else 0
    LB[dom] = 0
    for i in range(N_obs + N_vio):
        mask[:, i] = 1
        # dom[i, :] is "does i dominate :"
        m.addConstr(((mask - I) @ T) >= LB[i, :])  # T_i - T_j for all j * dom_ij
        mask[:, i] = 0

    m.setParam('OutputFlag', 1)
    m.optimize()

    info = {'lam': lam}

    if m.status == gp.GRB.OPTIMAL:
        dist = np.abs(delta_obs.x - np.abs(T.x[:N_obs] - Y_obs))
        print('Max delta dist:', np.max(dist), '; Total dist:', np.sum(dist))

        info['min_obj'] = m.objVal
        info['min_T_obj'] = np.sum(T.x[N_obs:])
        info['min_delta_obs'] = delta_obs.x
    else:
        info['min_status'] = m.status

    T.obj = -obj_T
    m.optimize()

    if m.status == gp.GRB.OPTIMAL:
        dist = np.abs(delta_obs.x - np.abs(T.x[:N_obs] - Y_obs))
        print('Max delta dist:', np.max(dist), '; Total dist:', np.sum(dist))

        info['max_obj'] = m.objVal
        info['max_T_obj'] = np.sum(T.x[N_obs:])
        info['max_delta_obs'] = delta_obs.x
    else:
        info['max_status'] = m.status

    min_obj = info['min_T_obj'] + np.sum((alloc * realScores)[alloc * hasScore > .5])
    min_obj /= np.sum(alloc)
    max_obj = info['max_T_obj'] + np.sum((alloc * realScores)[alloc * hasScore > .5])
    max_obj /= np.sum(alloc)

    return min_obj, max_obj


def compute_ymin_ymax_lipschitz(alloc, realScores, hasScore, userRep, estSimScores, kpScores, L_const):
    observed = np.where(hasScore > .5)
    obs_idx = []
    for i, j in zip(observed[0], observed[1]):
        obs_idx.append(i * alloc.shape[1] + j)
    alloc_but_not_obs = np.where((alloc > .5) & (hasScore < .5))
    vio_idx = []
    for i, j in zip(alloc_but_not_obs[0], alloc_but_not_obs[1]):
        vio_idx.append(i * alloc.shape[1] + j)

    X = np.column_stack((userRep.flatten(), estSimScores.flatten(), kpScores.flatten()))
    norm_zero_one(X)

    X_obs = X[obs_idx, :]
    X_vio = X[vio_idx, :]

    def cdist_lip(X1, X2):
        D = np.zeros((X1.shape[0], X2.shape[0]))
        for i in range(X1.shape[1]):
            Di = np.abs(np.subtract.outer(X1[:, i], X2[:, i]))
            Di[np.isnan(Di)] = 1
            D += Di
        D /= X1.shape[1]
        return D

    N_obs = int(np.sum(hasScore))
    N_vio = np.sum((hasScore < .5) & (alloc > .5))

    D_obs_obs = cdist_lip(X_obs, X_obs)
    D_vio_obs = cdist_lip(X_vio, X_obs)
    D_vio_vio = cdist_lip(X_vio, X_vio)
    D = np.zeros((N_obs + N_vio, N_obs + N_vio))
    D[:N_obs, :N_obs] = D_obs_obs
    D[N_obs:, N_obs:] = D_vio_vio
    D[N_obs:, :N_obs] = D_vio_obs
    D[:N_obs, N_obs:] = D_vio_obs.T

    m = gp.Model()
    m.setParam('OutputFlag', 1)
    m.setParam('Method', 1)

    obj_T = np.zeros(N_obs + N_vio)

    obj_T[N_obs:] = 1
    y_min = -5
    y_max = 1
    T = m.addMVar(N_obs + N_vio, lb=y_min, ub=y_max, obj=obj_T)

    lam = 1e9
    delta_obs = m.addMVar(N_obs, lb=0, ub=y_max - y_min, obj=lam)
    m.modelSense = gp.GRB.MINIMIZE

    Y_obs = realScores[hasScore > .5]
    m.addConstr(T[:N_obs] - Y_obs <= delta_obs)
    m.addConstr(Y_obs - T[:N_obs] <= delta_obs)

    mask = np.zeros((N_obs + N_vio, N_obs + N_vio))
    I = np.eye(N_obs + N_vio)
    for i in range(N_obs + N_vio):
        mask[:, i] = 1
        m.addConstr(((mask - I) @ T) <= L_const * D[:, i])
        mask[:, i] = 0

    m.optimize()

    info = {'lam': lam}

    if m.status == gp.GRB.OPTIMAL:
        dist = np.abs(delta_obs.x - np.abs(T.x[:N_obs] - Y_obs))
        print('Max delta dist:', np.max(dist), '; Total dist:', np.sum(dist))

        info['min_obj'] = m.objVal
        info['min_T_obj'] = np.sum(T.x[N_obs:])
        info['min_delta_obs'] = delta_obs.x
    else:
        info['min_status'] = m.status

    T.obj = -obj_T
    m.optimize()

    if m.status == gp.GRB.OPTIMAL:
        dist = np.abs(delta_obs.x - np.abs(T.x[:N_obs] - Y_obs))
        print('Max delta dist:', np.max(dist), '; Total dist:', np.sum(dist))

        info['max_obj'] = m.objVal
        info['max_T_obj'] = np.sum(T.x[N_obs:])
        info['max_delta_obs'] = delta_obs.x
    else:
        info['max_status'] = m.status

    min_obj = info['min_T_obj'] + np.sum((alloc * realScores)[alloc * hasScore > .5])
    min_obj /= np.sum(alloc)
    max_obj = info['max_T_obj'] + np.sum((alloc * realScores)[alloc * hasScore > .5])
    max_obj /= np.sum(alloc)

    return min_obj, max_obj


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

    for lam in [5]:
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
    metric_to_allocation_scores['lb_mono'] = {}
    metric_to_allocation_scores['ub_mono'] = {}
    metric_to_allocation_scores['lb_lip_1'] = {}
    metric_to_allocation_scores['ub_lip_1'] = {}
    metric_to_allocation_scores['lb_lip_2'] = {}
    metric_to_allocation_scores['ub_lip_2'] = {}
    metric_to_allocation_scores['lb_lip_3'] = {}
    metric_to_allocation_scores['ub_lip_3'] = {}

    # Level 1 is .01, level 2 is .05, level 3 is .1
    topic_to_lipschitz = {"cs": {1: 15.649, 2: 2.467, 3: 0.375},
                          "biology": {1: 16.744, 2: 4.642, 3: 1.993},
                          "chemistry": {1: 13.159, 2: 3.732, 3: 1.458},
                          "academia": {1: 21.880, 2: 5.683, 3: 2.310}}

    all_allocs = [pred_alloc, pred_alloc_user_embs, pred_alloc_badges] + non_pred_allocs + rand_allocs
    # all_names = ['pred', 'pred_user_embs', 'pred_badges'] + list(range(11)) + ["rand" + str(i) for i in range(1)]
    all_names = ['pred', 'pred_user_embs', 'pred_badges'] + [5] + ["rand" + str(i) for i in range(1)]

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

        lb_mono, ub_mono = \
            compute_ymin_ymax_monotonic(alloc, realScores, hasScore,
                                        user_rep_scores, estimated_topical_sim, kp_matching_scores)
        metric_to_allocation_scores['lb_mono'][alloc_name], metric_to_allocation_scores['ub_mono'][alloc_name] = \
            lb_mono, ub_mono

        pickle.dump(metric_to_allocation_scores,
                    open(os.path.join(data_dir, "metric_to_allocation_scores_%d.pkl" % seed), 'wb'))
        print("LB/UB mono for %s is %.2f, %.2f" % (alloc_name, lb_mono, ub_mono), flush=True)

        # for lip_level in [1, 2, 3]:
        for lip_level in [2]:
            lb_lip, ub_lip = \
                compute_ymin_ymax_lipschitz(alloc, realScores, hasScore,
                                            user_rep_scores, estimated_topical_sim,
                                            kp_matching_scores, topic_to_lipschitz[topic][lip_level])
            metric_to_allocation_scores['lb_lip_%d' % lip_level][alloc_name], metric_to_allocation_scores['ub_lip_%d' % lip_level][alloc_name] = \
                lb_lip, ub_lip

            pickle.dump(metric_to_allocation_scores,
                        open(os.path.join(data_dir, "metric_to_allocation_scores_%d.pkl" % seed), 'wb'))
            print("LB/UB lipschitz for %s at level %d is %.2f, %.2f" % (alloc_name, lip_level, lb_lip, ub_lip), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", type=str, default="cs")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    main(args)
