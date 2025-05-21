from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import sys
import torch
from tqdm import tqdm

# git clone https://github.com/fcumlin/DNSMOSPro
dnsmos_pro_dir = "./DNSMOSPro"
sys.path.append(dnsmos_pro_dir)
import utils


METRICS = ("DNSMOSPro",)
TARGET_FS = 16000


def str2bool(value: str) -> bool:
    val = value.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError("invalid truth value %r" % (val,))


################################################################
# Definition of metrics
################################################################
def dnsmos_pro_metric(model, audio, fs=16000, device="cpu"):
    """Calculate the DNSMOS Pro metric.

    Reference:
        Fredrik Cumlin, Xinyu Liang, Victor Ungureanu, Chandan K. A. Reddy,
        Christian Schüldt, and Saikat Chatterjee.
        DNSMOS Pro: A Reduced-Size DNN for Probabilistic MOS of Speech.
        in Proc. Interspeech 2024, pp. 4818-4822.

    Args:
        model (torch.nn.Module): DNSMOS Pro model
        audio (np.ndarray): enhanced signal (time,)
        fs (int): sampling rate in Hz
    Returns:
        mos_score (float): predicted MOS value between [1, 5]
    """
    if fs != TARGET_FS:
        audio = soxr.resample(audio, fs, TARGET_FS)
        fs = TARGET_FS
    with torch.no_grad():
        spec = torch.FloatTensor(utils.stft(audio)).to(device=device)
        prediction = model(spec[None, None, ..., ])
    mean = prediction[:, 0].cpu().item()
    # variance = prediction[:, 1].cpu().item()
    return mean


################################################################
# Main entry
################################################################
def main(args):
    data_pairs = []
    with open(args.inf_scp, "r") as f:
        for line in f:
            uid, audio_path = line.strip().split()
            data_pairs.append((uid, audio_path))

    size = len(data_pairs)
    assert 1 <= args.job <= args.nsplits <= size
    interval = size // args.nsplits
    start = (args.job - 1) * interval
    end = size if args.job == args.nsplits else start + interval
    data_pairs = data_pairs[start:end]
    print(
        f"[Job {args.job}/{args.nsplits}] Processing ({len(data_pairs)}/{size}) samples",
        flush=True,
    )
    suffix = "" if args.nsplits == args.job == 1 else f".{args.job}"

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    writers = {
        metric: (outdir / f"{metric}{suffix}.scp").open("w") for metric in METRICS
    }

    model = torch.jit.load(args.model_path, map_location=torch.device(args.device))
    model.eval()
    ret = []
    for uid, inf_audio in tqdm(data_pairs):
        _, score = process_one_pair((uid, inf_audio), model=model, device=args.device)
        ret.append((uid, score))
        for metric, value in score.items():
            writers[metric].write(f"{uid} {value}\n")

    for metric in METRICS:
        writers[metric].close()

    if args.nsplits == args.job == 1:
        with (outdir / "RESULTS.txt").open("w") as f:
            for metric in METRICS:
                mean_score = np.nanmean([score[metric] for uid, score in ret])
                f.write(f"{metric}: {mean_score:.4f}\n")
        print(
            f"Overall results have been written in {outdir / 'RESULTS.txt'}", flush=True
        )


def process_one_pair(data_pair, model=None, device="cpu"):
    uid, inf_path = data_pair
    inf, fs = sf.read(inf_path, dtype="float32")
    assert inf.ndim == 1, inf.shape

    scores = {}
    for metric in METRICS:
        if metric in scores:
            continue
        if metric in ("DNSMOSPro",):
            scores[metric] = dnsmos_pro_metric(model, inf, fs=fs, device=device)
        else:
            raise NotImplementedError(metric)

    return uid, scores


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inf_scp",
        type=str,
        required=True,
        help="Path to the scp file containing enhanced signals",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory for writing metrics",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for running DNSMOS Pro calculation",
    )
    parser.add_argument(
        "--nsplits",
        type=int,
        default=1,
        help="Total number of computing nodes to speed up evaluation",
    )
    parser.add_argument(
        "--job",
        type=int,
        default=1,
        help="Index of the current node (starting from 1)",
    )

    group = parser.add_argument_group("DNSMOS Pro related")
    group.add_argument(
        "--model_path",
        type=str,
        default="DNSMOSPro/runs/NISQA/model_best.pt",
        help="Path to the pretrained DNSMOS Pro model.",
    )
    args = parser.parse_args()

    main(args)
