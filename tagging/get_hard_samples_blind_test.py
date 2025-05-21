import json
import re

from math import nan
from pathlib import Path

import numpy as np
import scipy.stats as ss


METRIC_CATEGORIES = (
    # Non-intrusive SE metrics
    ("DNSMOS_OVRL", "NISQA_MOS"),
    # Intrusive SE metrics
    ("POLQA", "PESQ", "ESTOI", "SDR", "MCD", "LSD"),
    # ("PESQ", "ESTOI", "SDR", "MCD", "LSD"),
    # Downstream-task-independent metrics
    ("SpeechBERTScore", "PhonemeSimilarity"),
    # Downstream-task-dependent metrics
    ("SpeakerSimilarity", "WER"),
    # Subjective SE metrics
    ("MOS",),
)
METRICS, WEIGHTS = zip(
    *[
        (k, 1.0 / len(METRIC_CATEGORIES) / len(tup))  # more consistent with CodaLab
        # (k, Fraction(1, len(METRIC_CATEGORIES) * len(tup)))
        for tup in METRIC_CATEGORIES
        for k in tup
    ]
)
WEIGHTS = np.array(WEIGHTS)

# Samples with a score larger than the threshold are of poor quality
METRIC_THRESHOLDS = {
    "DNSMOS_OVRL": -2.0,
    "NISQA_MOS": -2.0,
    "POLQA": -1.7,
    "PESQ": -1.5,
    "ESTOI": -0.6,
    "SDR": -0.0,
    "MCD": 5.0,
    "LSD": 5.0,
    "SpeechBERTScore": -0.5,
    "PhonemeSimilarity": -0.4,
    "SpeakerSimilarity": -0.4,
    "WER": 0.5,
    "MOS": -2.0,
}
assert len(METRIC_THRESHOLDS) == len(METRICS)


PID2NAME = {
    "422": "FSInfo",
    "426": "noisy",
    "427": "urgent",
    "455": "ww",
    "456": "VenkateshP",
    "474": "TeamDUT",
    "477": "N-B",
    "478": "Xiaobin",
    "481": "rc",
    "488": "RICK2000",
    "489": "Hanchen_Pei",
    "492": "oceane",
    "495": "Luo",
    "505": "Bobbsun",
    "506": "Enhance",
    "508": "fliu",
    "509": "Julius_Richter",
    "511": "honee",
    "518": "haeon",
    "523": "WHU-NERCMS",
    "520": "inverseai",
    "524": "Linfeng_Feng",
    "526": "byti.shsy",
}

SUBSET = None


# def relative_comparison(uids, pids, scores, uidCounter={}):
#     # Step 1: Get scores of original input speech
#     for p in PID2NAME:
#         if p in pids:
#             index = pids.index(p)
#             break
#     else:
#         raise ValueError("Original input speech PID not found")
#     org_scores = {m: scores[m][index] for m in METRICS}

#     # Step 2: Find samples where the (weighted-sum) enhanced score is worse than the original score
#     for i, pid in enumerate(pids):
#         if pid == p:
#             continue
#         for j, uid in enumerate(uids):
#             rate = []
#             for k, metric in enumerate(METRICS):
#                 if np.isnan(scores[metric][i, j]) or np.isnan(org_scores[metric][j]):
#                     rate.append(0)
#                 elif scores[metric][i, j] > org_scores[metric][j]:
#                     # print(f"pid={pid}, uid={uid}, {metric}: {scores[metric][i, j]} > {org_scores[metric][j]}")
#                     rate.append(-WEIGHTS[k])
#                 else:
#                     rate.append(WEIGHTS[k])
#             if sum(rate) < 0:
#                 uidCounter.setdefault(uid, set()).add(pid)
#     return uidCounter


def relative_comparison(uids, pids, scores, uidCounter={}, verbose=False):
    # Step 1: Get scores of original input speech
    for p in PID2NAME:
        if p in pids:
            index = pids.index(p)
            break
    else:
        raise ValueError("Original input speech PID not found")

    # Step 2: Calculate overall ranking score for each sample (uid) in each team (pid)
    # (n_samples, n_submissions)
    overall_ranking_score = np.stack(
        [
            get_ranking(np.stack([scores[m][:, j] for m in METRICS], axis=1))[1]
            for j in range(len(uids))
        ],
        axis=0,
    )

    # Step 3: Find samples where the overall ranking score is higher than the original input's score
    for i, pid in enumerate(pids):
        if pid == p:
            continue
        for j, uid in enumerate(uids):
            if verbose:
                msg = f"------------- pid={pid}, uid={uid} -------------\n"
            if overall_ranking_score[j, i] > overall_ranking_score[j, index]:
                uidCounter.setdefault(uid, set()).add(pid)
                if verbose:
                    msg += f"overall_ranking_score: {overall_ranking_score[j, i]} > {overall_ranking_score[j, index]}"
                    print(msg)
    return uidCounter


def absolute_comparison(uids, pids, scores, uidCounter={}, verbose=False):
    # Step 1: Find the index of the original input speech
    for p in PID2NAME:
        if p in pids:
            break
    else:
        raise ValueError("Original input speech PID not found")

    # Step 2: Find samples where the (weighted-sum) enhanced score is worse than the threshold
    for i, pid in enumerate(pids):
        if pid == p:
            continue
        for j, uid in enumerate(uids):
            rate = []
            if verbose:
                msg = f"------------- pid={pid}, uid={uid} -------------\n"
            for k, metric in enumerate(METRICS):
                if np.isnan(scores[metric][i, j]):
                    rate.append(0)
                elif scores[metric][i, j] > METRIC_THRESHOLDS[metric]:
                    if verbose:
                        msg += f"{metric}: {scores[metric][i, j]} > {METRIC_THRESHOLDS[metric]}\n"
                    rate.append(-WEIGHTS[k])
                else:
                    rate.append(WEIGHTS[k])
            if sum(rate) < 0:
                if verbose:
                    print(msg)
                uidCounter.setdefault(uid, set()).add(pid)
    return uidCounter


def load(submission_dir):
    uids = set()
    pids = []
    for sub in Path(submission_dir).iterdir():
        if not sub.is_dir() or sub.stem in (".DS_Store", "__MACOSX"):
            continue
        pids.append(sub.stem)
        for scp in sub.rglob("*.scp"):
            metric = scp.stem
            if re.match(r".*\.\d+", metric):
                continue
            with scp.open("r") as f:
                for line in f:
                    uid, value = line.rstrip().split(maxsplit=1)
                    uids.add(uid)
    if SUBSET:
        uids = sorted(SUBSET, key=lambda x: int(x.split("_", maxsplit=1)[1]))
    else:
        uids = sorted(uids, key=lambda x: int(x.split("_", maxsplit=1)[1]))
    pids = sorted(pids)

    scores = {m: {pid: {uid: nan for uid in uids} for pid in pids} for m in METRICS}
    for sub in Path(submission_dir).iterdir():
        if not sub.is_dir() or sub.stem in (".DS_Store", "__MACOSX"):
            continue
        pid = sub.stem
        for scp in sub.rglob("*.scp"):
            metric = scp.stem
            if re.match(r".*\.\d+", metric):
                continue
            with scp.open("r") as f:
                for line in f:
                    uid, value = line.rstrip().split(maxsplit=1)
                    if metric == "WER":
                        d = json.loads(value)
                        numerator = d["replace"] + d["delete"] + d["insert"]
                        denominator = d["replace"] + d["delete"] + d["equal"]
                        scores[metric][pid][uid] = numerator / denominator
                    else:
                        scores[metric][pid][uid] = float(value)

        for m in METRICS:
            scores[m][pid] = np.array([scores[m][pid][uid] for uid in uids])

        avg_score = []
        for txt in sub.rglob("RESULTS.txt"):
            with txt.open("r") as f:
                for line in f:
                    metric, value = line.rstrip().split(":")
                    if metric.startswith((" ", "WER")):
                        continue
                    avg_score = average_metric(metric, scores[metric][pid])
                    if not SUBSET:
                        assert float(avg_score) == float(value), (avg_score, line)

    for m in METRICS:
        # scores[m].shape: (n_submissions, n_samples)
        scores[m] = unify_for_comparison(m, np.array(list(scores[m].values())))
    return uids, pids, scores


def unify_for_comparison(metric, values):
    return -values if metric not in ("MCD", "LSD", "WER") else values


def average_metric(metric, arr, decimals=4):
    ret = np.nanmean(arr, axis=-1)
    if ret.size == 1:
        return np.round(ret.item(), decimals=decimals)
    return np.array([n for n in np.round(ret, decimals=decimals)])


def get_ranking(avg_scores):
    """Calculate the overall ranking based on average scores of each metric.

    Args:
        avg_scores (array): average metric scores (n_submissions, n_metrics)
            The metrics are assumed to be ordered according to METRICS.
    Returns:
        per_metric_ranking (array): ranking of each submission for each metric
            (n_submissions, n_metrics)
        overall_ranking_score (array): overall ranking score of each submission
            (n_submissions,)
    """
    if np.array([np.isnan(i) for i in avg_scores.flatten()]).any():
        avg_scores = np.nan_to_num(avg_scores, copy=False)  # Convert NaN to 0.0
    per_metric_ranking = ss.rankdata(avg_scores, method="dense", axis=0)
    return per_metric_ranking, np.nansum(per_metric_ranking * WEIGHTS[None, :], axis=1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--submission_dir",
        type=str,
        default="submissions_mos",
        help="Path to the root directory containing all submissions",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default="output.scp",
        help="Output scp flie for storing all results",
    )
    parser.add_argument(
        "--ndigits",
        type=int,
        default=2,
        help="Number of digits after the decimal",
    )
    args = parser.parse_args()

    uids, pids, scores = load(args.submission_dir)

    uidCounter = {}
    uidCounter = absolute_comparison(uids, pids, scores, uidCounter=uidCounter)
    print(f"Absolute strategy: {len(uidCounter)} poor sample UIDs found among toally {len(uids)} UIDs.")
    print(f"    {sum(len(v) for v in uidCounter.values())} poor samples among totally {len(uids) * (len(pids) - 1)} samples")
    # print(sorted([len(v) for uid, v in uidCounter.items()], key=lambda x: -x))
    # for uid, count in sorted([(uid, len(v)) for uid, v in uidCounter.items()], key=lambda x: -x[1]):
    #     if count > 3:
    #         print(f"   {uid}: {count} poor samples")
    with open(args.outfile, "w") as f:
        for uid, _pids in uidCounter.items():
            f.write(f"{uid} {','.join(_pids)}\n")

    uidCounter = relative_comparison(uids, pids, scores, uidCounter=uidCounter)
    print(f"Relative strategy: {len(uidCounter)} poor sample UIDs found among toally {len(uids)} UIDs.")
    print(f"    {sum(len(v) for v in uidCounter.values())} poor samples among totally {len(uids) * (len(pids) - 1)} samples")
    # print(sorted([len(v) for uid, v in uidCounter.items()], key=lambda x: -x))
    # for uid, count in sorted([(uid, len(v)) for uid, v in uidCounter.items()], key=lambda x: -x[1]):
    #     if count > 3:
    #         print(f"   {uid}: {count} poor samples")
    with open(args.outfile, "w") as f:
        for uid, _pids in uidCounter.items():
            f.write(f"{uid} {','.join(_pids)}\n")
