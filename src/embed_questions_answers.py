from xml.etree import ElementTree as ET
import argparse
import pickle
import re
import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util


def remove_tabs(x):
    y = re.sub(r"\s+", " ", x)
    z = re.sub(r"[\'\"\\\\]", "", y)
    q = re.sub(r"\n", " ", z)
    return q


def main(args):
    topic = args.topic

    model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-cos-v1')

    base_dir = "/work/pi_yzick_umass_edu/jpayan/predictive_expert_assignment"

    tree = ET.parse(os.path.join(base_dir, "data/%s.stackexchange.com/Posts.xml" % topic))
    root = tree.getroot()
    posts = []
    for child in root:
        posts.append(child.attrib)
    id_to_title_and_body = {}

    for p in posts:
        if 'Title' in p and 'Body' in p:
            id_to_title_and_body[p['Id']] = {'Title': p['Title'], 'Body': p['Body']}
        elif 'Body' in p:
            id_to_title_and_body[p['Id']] = {'Body': p['Body']}

    print("Posts loaded, beginning embedding", flush=True)

    # Sort the ids and write that out
    # Also write out a list of the ids that specifically have titles, so we can index those embeddings easier.
    sorted_all_ids = sorted(list(id_to_title_and_body.keys()))
    sorted_has_title = sorted([x for x in id_to_title_and_body.keys() if 'Title' in id_to_title_and_body[x]])

    with open(os.path.join(base_dir, "output/%s_all_ids.pkl" % topic), 'wb') as f:
        pickle.dump(sorted_all_ids, f)
    with open(os.path.join(base_dir, "output/%s_ids_for_titles.pkl" % topic), 'wb') as f:
        pickle.dump(sorted_has_title, f)

    # Now compute the embeddings
    all_titles = [id_to_title_and_body[x]['Title'] for x in sorted_has_title]
    all_bodies = [id_to_title_and_body[x]['Body'] for x in sorted_all_ids]

    title_embs = model.encode(all_titles, convert_to_tensor=True)
    body_embs = model.encode(all_bodies, convert_to_tensor=True)

    # Write out embeddings
    np.save("output/%s_title_embs.npy" % topic, title_embs.detach().numpy())
    np.save("output/%s_body_embs.npy" % topic, body_embs.detach().numpy())

    # Compute cosine similarities ahead of time
    cos_sims_title_title = util.cos_sim(title_embs, title_embs)[0]
    np.save("output/%s_cos_sims_title_title.npy" % topic, cos_sims_title_title.detach().numpy())

    cos_sims_title_body = util.cos_sim(title_embs, body_embs)[0]
    np.save("output/%s_cos_sims_title_body.npy" % topic, cos_sims_title_body.detach().numpy())

    cos_sims_body_body = util.cos_sim(body_embs, body_embs)[0]
    np.save("output/%s_cos_sims_body_body.npy" % topic, cos_sims_body_body.detach().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", type=str, default="cs")

    args = parser.parse_args()
    main(args)
