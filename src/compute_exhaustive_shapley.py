import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
import pickle
import torch
from torch import nn
import os


def get_feat_idxs(feats_selected, feat_list):
    idxs = []
    for f in feats_selected:
        idxs.append(feat_list.index(f))
    return np.array(idxs)


def compute_cross_ent_loss(selected_feat_names, feat_list, train_set, train_evs, test_set, test_evs, xe_test_constant,
                           max_iter=100):
    loss_fn = nn.CrossEntropyLoss()

    if len(selected_feat_names):
        # Get the right features out
        feat_idxs = get_feat_idxs(selected_feat_names, feat_list)
        x_feats = np.array(train_set)[:, feat_idxs]

        # Renormalize
        n1 = np.max(x_feats, axis=0)
        normalizer_scores = np.max(np.linalg.norm(x_feats / n1, axis=1))
        scaler_votes = lambda y: (y / n1) / normalizer_scores
        x_feats = scaler_votes(x_feats)
        ylogreg = train_evs

        dups = []
        labels = []
        class_weights = []
        for idx in range(x_feats.shape[0]):
            feats, probs = x_feats[idx, :], ylogreg[idx]
            dups.extend([feats] * 2)
            labels.extend([0, 1])
            class_weights.extend(probs)

        test_x_feats = scaler_votes(np.array(test_set)[:, feat_idxs])

        # Set up the model
        log_reg = LogisticRegression()
        log_reg.fit(dups, labels, class_weights)

        # Compute the dev and test XE
        predicted_prob_upvote_test = torch.tensor(log_reg.predict_log_proba(test_x_feats))

        xe_test = loss_fn(predicted_prob_upvote_test, test_evs)
        return 0.0, xe_test.item()
    else:
        return 0.0, xe_test_constant.item()


def main(args):
    topic = args.topic
    base_dir = "/mnt/nfs/scratch1/jpayan/predictive_expert_assignment"

    # First, open up the train and test sets, as well as the evs you're supposed to predict for train and test,
    # and the constant predictor
    data_dir = os.path.join(base_dir, "data", "%s.stackexchange.com" % topic)
    with open(os.path.join(data_dir, "train_set.pkl"), 'rb') as f:
        train_set = pickle.load(f)
    with open(os.path.join(data_dir, "test_set.pkl"), 'rb') as f:
        test_set = pickle.load(f)
    with open(os.path.join(data_dir, "train_evs.pkl"), 'rb') as f:
        train_evs = pickle.load(f)
    with open(os.path.join(data_dir, "test_evs.pkl"), 'rb') as f:
        test_evs = pickle.load(f)
    with open(os.path.join(data_dir, "xe_test_constant.pkl"), 'rb') as f:
        xe_test_constant = pickle.load(f)

    feat_list = ["Reputation", "views", "upvotes", "downvotes", "keyword match score", "avg time to answer", "MRR",
                 "avg of views", "avg of scores", "smoothed frac best answers", "use", "rel", "inf",
                 "mean cos sim titles", "max cos sim titles", "mean cos sim bodies", "max cos sim bodies",
                 "past_scores"]
    full_feat_list = ["Reputation", "views", "upvotes", "downvotes", "keyword match score", "avg time to answer", "MRR",
                 "avg of views", "avg of scores", "smoothed frac best answers", "use", "rel", "inf",
                 "mean cos sim titles", "max cos sim titles", "mean cos sim bodies", "max cos sim bodies"]
    full_feat_list.extend("score%d" % s for s in [0, 5, 10, 25, 50])

    num_feats = len(feat_list)
    test_xe_dict = {}

    for feat_selector in range(2**num_feats):
        if feat_selector % 1000 == 0:
            print("Processed %d of %d" % (feat_selector, 2**num_feats), flush=True)
            pickle.dump(test_xe_dict, open(os.path.join(data_dir, "test_xe_dict.pkl"), 'wb'))

        bin_feat_sel = str(bin(feat_selector))[2:]
        bin_feat_sel = '0'*(num_feats - len(bin_feat_sel)) + bin_feat_sel

        chosen = []
        for idx, onoff in enumerate(bin_feat_sel):
            if onoff == '1':
                chosen.append(feat_list[idx])
        if 'past_scores' in chosen:
            chosen.remove('past_scores')
            chosen.extend("score%d" % s for s in [0, 5, 10, 25, 50])

        _, test_xe = compute_cross_ent_loss(chosen, full_feat_list,
                                            train_set, train_evs, test_set,
                                            test_evs, xe_test_constant)
        test_xe_dict[feat_selector] = test_xe
    pickle.dump(test_xe_dict, open(os.path.join(data_dir, "test_xe_dict.pkl"), 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", type=str, default="cs")

    args = parser.parse_args()
    main(args)
