from xml.etree import ElementTree as ET
import argparse
import pickle
import re
import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

def remove_tabs(x):
    y = re.sub(r"\s+", " ", x)
    z = re.sub(r"[\'\"\\\\]", "", y)
    q = re.sub(r"\n", " ", z)
    return q


def main(args):
    topic = args.topic

    base_dir = "/work/pi_yzick_umass_edu/jpayan/predictive_expert_assignment"

    # Load embeddings
    title_embs = np.load(os.path.join(base_dir, "output/%s/title_embs.npy" % topic))
    body_embs = np.load(os.path.join(base_dir, "output/%s/body_embs.npy" % topic))

    # Compute cosine similarities ahead of time
    cos_sims_title_title = cosine_similarity(title_embs)
    np.save(os.path.join(base_dir, "output/%s/cos_sims_title_title.npy" % topic), cos_sims_title_title)

    cos_sims_title_body = cosine_similarity(title_embs, body_embs)
    np.save(os.path.join(base_dir, "output/%s/cos_sims_title_body.npy" % topic), cos_sims_title_body)

    cos_sims_body_body = cosine_similarity(body_embs)
    np.save(os.path.join(base_dir, "output/%s/cos_sims_body_body.npy" % topic), cos_sims_body_body)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", type=str, default="cs")

    args = parser.parse_args()
    main(args)
