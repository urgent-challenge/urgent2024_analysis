import yaml
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import sys
import torch
from tqdm import tqdm

# git clone https://github.com/JasonSWFu/VQscore
vqscore_dir = "./VQscore"
sys.path.append(vqscore_dir)
from models.VQVAE_models import VQVAE_SE, VQVAE_QE


METRICS = ("VQscore",)
TARGET_FS = 16000


# ported from VQscore/inference.py
def stft_magnitude(x, hop_size, fft_size=512, win_length=512):
    if x.is_cuda:
        x_stft = torch.stft(
            x,
            fft_size,
            hop_size,
            win_length,
            window=torch.hann_window(win_length).to("cuda"),
            return_complex=False,
        )
    else:
        x_stft = torch.stft(
            x,
            fft_size,
            hop_size,
            win_length,
            window=torch.hann_window(win_length),
            return_complex=False,
        )
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    return torch.sqrt(torch.clamp(real**2 + imag**2, min=1e-7)).transpose(2, 1)


def cos_similarity(SP_noisy, SP_y_noisy, eps=1e-5):
    SP_noisy_norm = torch.norm(SP_noisy, p=2, dim=-1, keepdim=True) + eps
    SP_y_noisy_norm = torch.norm(SP_y_noisy, p=2, dim=-1, keepdim=True) + eps
    Cos_frame = torch.sum(
        SP_noisy / SP_noisy_norm * SP_y_noisy / SP_y_noisy_norm, dim=-1
    )  # torch.Size([B, T, 1]

    return torch.mean(Cos_frame)


################################################################
# Definition of metrics
################################################################
def vqscore_metric(model, audio, fs=16000, hop_size=256, device="cpu"):
    """Calculate the VQscore metric.

    Reference:
        Szu-Wei Fu, Kuo-Hsuan Hung, Yu Tsao, and Yu-Chiang Frank Wang.
        Self-Supervised Speech Quality Estimation and Enhancement Using Only
        Clean Speech. in ICLR 2024.

    Args:
        model (torch.nn.Module): VQscore model
        audio (np.ndarray): enhanced signal (time,)
        fs (int): sampling rate in Hz
        hop_size (int): hop size for STFT
    Returns:
        vqscore (float): predicted VQScore value between [-1.0, 1.0]
    """
    if fs != TARGET_FS:
        audio = soxr.resample(audio, fs, TARGET_FS)
        fs = TARGET_FS
    with torch.no_grad():
        audio = torch.from_numpy(audio).to(device=device).unsqueeze(0)
        SP_input = stft_magnitude(audio, hop_size=hop_size)
        if model.input_transform == "log1p":
            SP_input = torch.log1p(SP_input)
        z = model.CNN_1D_encoder(SP_input)
        zq, indices, vqloss, distance = model.quantizer(
            z, stochastic=False, update=False
        )
        SP_output = model.CNN_1D_decoder(zq)

        VQScore_cos_z = cos_similarity(z.transpose(2, 1).cpu(), zq.cpu()).numpy()

    return VQScore_cos_z


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

    with open(args.vqscore_conf, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if args.device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
    model = VQVAE_QE(**config["VQVAE_params"]).to(device=args.device).eval()
    model.load_state_dict(torch.load(args.vqscore_model)["model"]["VQVAE"])
    model.input_transform = config["input_transform"]
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
        if metric in ("VQscore",):
            scores[metric] = vqscore_metric(model, inf, fs=fs, device=device)
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
        help="Device for running VQscore calculation",
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

    group = parser.add_argument_group("VQscore related")
    group.add_argument(
        "--vqscore_conf",
        type=str,
        default="VQscore/config/QE_cbook_size_2048_1_32_IN_input_encoder_z_"
        "Librispeech_clean_github.yaml",
        help="Path to the VQscore model configuration.",
    )
    group.add_argument(
        "--vqscore_model",
        type=str,
        default="VQscore/exp/QE_cbook_size_2048_1_32_IN_input_encoder_z_"
        "Librispeech_clean_github/checkpoint-dnsmos_ovr_CC=0.835.pkl",
        help="Path to the pretrained VQscore model.",
    )
    args = parser.parse_args()

    main(args)
