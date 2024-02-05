from xml.etree import ElementTree as ET
from collections import defaultdict
import subprocess
import threading
import argparse
import time
import re
import csv
import os


def run_controller():
    subprocess.run(["python3", "-m", "fastchat.serve.controller", "--host", "127.0.0.1"])


def run_model_worker():
    subprocess.run(["python3", "-m", "fastchat.serve.model_worker", "--host", "127.0.0.1", "--controller-address",
                    "http://127.0.0.1:21001", "--model-path", "lmsys/vicuna-7b-v1.5", "--load-8bit"])


def run_api_server():
    subprocess.run(["python3", "-m", "fastchat.serve.openai_api_server", "--host", "127.0.0.1", "--controller-address",
                    "http://127.0.0.1:21001", "--port", "8000"])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", type=str, default="cs")
    parser.add_argument("--feat_type", type=str, default="answer_quality")

    return parser.parse_args()


def remove_tabs(x):
    y = re.sub(r"\s+", " ", x)
    z = re.sub(r"[\'\"\\\\]", "", y)
    q = re.sub(r"\n", " ", z)
    return q


def build_query_answer_feats(title, body, ans, topic):
    q = "I am going to provide you with a question-answer pair from the %s StackExchange. " % topic
    q += "Please annotate the informativeness, relevance, and usefulness of the answer. "
    q += "Your response should rate each of these three aspects on a scale from 1-5, with 1 being the least and 5 being the most. "
    q += "Please structure your response by outputting the informativeness, then the relevance, and then the usefulness, one per line. "
    q += "Please do not include any additional explanation of your ratings. "
    q += "Informativeness asks Does this answer provide enough information for the question? "
    q += "Relevance asks Is this answer relevant to the question? "
    q += "Usefulness asks Is this answer useful or helpful to address the question?\\n\\n"
    q += "An example output might look like: \\nInformativeness: <Rating>\\nRelevance: <Rating>\\nUsefulness: <Rating>"
    content = "Question:\\nTitle: %s\\n\\nBody: %s\\n\\nAnswer:\\n%s\\n" % (title, body, ans)
    return q, content


if __name__ == "__main__":
    controller_thread = threading.Thread(target=run_controller)
    controller_thread.start()

    model_worker_thread = threading.Thread(target=run_model_worker)
    model_worker_thread.start()

    api_server_thread = threading.Thread(target=run_api_server)
    api_server_thread.start()

    args = parse_args()
    topic = args.topic
    feat_type = args.feat_type

    base_dir = "/work/pi_yzick_umass_edu/jpayan/predictive_expert_assignment"

    if feat_type == "answer_quality":
        tree = ET.parse(os.path.join(base_dir, "data/%s.stackexchange.com/Posts.xml" % topic))
        root = tree.getroot()
        posts = []
        for child in root:
            posts.append(child.attrib)
        questions = {}
        question_id_to_answers = defaultdict(list)
        pid_to_qid = {}

        for p in posts:
            if p['PostTypeId'] == '1' and int(p['Score']) > 2:
                questions[p['Id']] = p
            elif p['PostTypeId'] == '2':
                question_id_to_answers[p['ParentId']].append(p)
                pid_to_qid[p['Id']] = p['ParentId']

        with open(os.path.join(base_dir, "output/%s_%s_annotations.tsv" % (topic, feat_type)), 'w', encoding='utf-8') as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["QID", "AID", "Informativeness", "Relevance", "Usefulness"])
            for qid, answers in question_id_to_answers.items():
                for a in answers:
                    aid = a['Id']
                    try:
                        title = remove_tabs(questions[qid]['Title'])
                        body = remove_tabs(questions[qid]['Body'])
                        ans = remove_tabs(a['Body'])
                        query, content = build_query_answer_feats(title, body, ans, topic)
                        llm_query_str = '{"model": "vicuna-7b-v1.5", "messages": [{"role": "system", "content": "%s"}, ' \
                                        '{"role": "user", "content": "%s"}]}' % (
                                            query, content)
                        print(llm_query_str)
                        os.system('curl http://127.0.0.1:8000/v1/chat/completions \
                        -H "Content-Type: application/json" \
                        -d \'' + llm_query_str + ("\' > tmp"))
                        with open("tmp", 'r') as f:
                            response = eval(f.read())
                            full_response = response["choices"][0]["message"]["content"]
                            informativeness = int(re.search("Informativeness: ([1-5])", full_response)[1])
                            relevance = int(re.search("Relevance: ([1-5])", full_response)[1])
                            usefulness = int(re.search("Usefulness: ([1-5])", full_response)[1])
                    except:
                        informativeness, relevance, usefulness = -1, -1, -1
                    w.writerow([qid, aid, informativeness, relevance, usefulness])

    elif feat_type == "post_similarity":
        pass
