from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm.contrib.concurrent import process_map


METRICS = ("WADASNR",)


integral_lookup_table = {
    "-20 dB": 0.409747739,
    "-19 dB": 0.409869263,
    "-18 dB": 0.409985656,
    "-17 dB": 0.409690892,
    "-16 dB": 0.409861864,
    "-15 dB": 0.409990055,
    "-14 dB": 0.410271377,
    "-13 dB": 0.410526266,
    "-12 dB": 0.411010238,
    "-11 dB": 0.411432644,
    "-10 dB": 0.412317178,
    "-9 dB": 0.413372716,
    "-8 dB": 0.415264259,
    "-7 dB": 0.417819198,
    "-6 dB": 0.420772515,
    "-5 dB": 0.424527992,
    "-4 dB": 0.429188858,
    "-3 dB": 0.435103734,
    "-2 dB": 0.442341951,
    "-1 dB": 0.451614855,
    "0 dB": 0.462211529,
    "1 dB": 0.474916474,
    "2 dB": 0.488838093,
    "3 dB": 0.505092356,
    "4 dB": 0.523537093,
    "5 dB": 0.543720882,
    "6 dB": 0.565324274,
    "7 dB": 0.588475317,
    "8 dB": 0.613462118,
    "9 dB": 0.639544959,
    "10 dB": 0.667508177,
    "11 dB": 0.695837243,
    "12 dB": 0.724547622,
    "13 dB": 0.754147993,
    "14 dB": 0.783231484,
    "15 dB": 0.81240985,
    "16 dB": 0.842197752,
    "17 dB": 0.871664058,
    "18 dB": 0.900305039,
    "19 dB": 0.928804177,
    "20 dB": 0.95655449,
    "21 dB": 0.983534905,
    "22 dB": 1.010471548,
    "23 dB": 1.0362095,
    "24 dB": 1.061364248,
    "25 dB": 1.085793118,
    "26 dB": 1.109481904,
    "27 dB": 1.132779949,
    "28 dB": 1.154728256,
    "29 dB": 1.176273084,
    "30 dB": 1.197035028,
    "31 dB": 1.216716938,
    "32 dB": 1.235358982,
    "33 dB": 1.253643127,
    "34 dB": 1.271038908,
    "35 dB": 1.287180295,
    "36 dB": 1.303028647,
    "37 dB": 1.318395272,
    "38 dB": 1.332948173,
    "39 dB": 1.347009353,
    "40 dB": 1.360572696,
    "41 dB": 1.373455135,
    "42 dB": 1.385771224,
    "43 dB": 1.397335037,
    "44 dB": 1.408563968,
    "45 dB": 1.41959619,
    "46 dB": 1.42983624,
    "47 dB": 1.439584667,
    "48 dB": 1.449021764,
    "49 dB": 1.458048307,
    "50 dB": 1.466695685,
    "51 dB": 1.474869384,
    "52 dB": 1.48269965,
    "53 dB": 1.490343394,
    "54 dB": 1.49748214,
    "55 dB": 1.504351061,
    "56 dB": 1.510764265,
    "57 dB": 1.516989146,
    "58 dB": 1.522909703,
    "59 dB": 1.528578001,
    "60 dB": 1.533898351,
    "61 dB": 1.539121095,
    "62 dB": 1.543906502,
    "63 dB": 1.54858517,
    "64 dB": 1.553107762,
    "65 dB": 1.557443906,
    "66 dB": 1.561649273,
    "67 dB": 1.565663481,
    "68 dB": 1.569386712,
    "69 dB": 1.573077668,
    "70 dB": 1.576547638,
    "71 dB": 1.57980083,
    "72 dB": 1.583041292,
    "73 dB": 1.586024961,
    "74 dB": 1.588806813,
    "75 dB": 1.591624771,
    "76 dB": 1.594196895,
    "77 dB": 1.596931549,
    "78 dB": 1.599446005,
    "79 dB": 1.601850111,
    "80 dB": 1.604086681,
    "81 dB": 1.60627134,
    "82 dB": 1.608261987,
    "83 dB": 1.610045475,
    "84 dB": 1.611924722,
    "85 dB": 1.61369656,
    "86 dB": 1.615340743,
    "87 dB": 1.616889049,
    "88 dB": 1.618389159,
    "89 dB": 1.619853744,
    "90 dB": 1.621358779,
    "91 dB": 1.622681189,
    "92 dB": 1.623904229,
    "93 dB": 1.625131432,
    "94 dB": 1.626324628,
    "95 dB": 1.6274027,
    "96 dB": 1.628427675,
    "97 dB": 1.629455321,
    "98 dB": 1.6303307,
    "99 dB": 1.631280263,
    "100 dB": 1.632041021,
}
dbvals = np.array([float(x.split()[0]) for x in integral_lookup_table.keys()])
Gvals = np.array(list(integral_lookup_table.values()))


def wada_snr(audio, min_val=1e-10):
    """WADA-SNR (Waveform Amplitude Distribution Analysis) algorithm.

    Assume that the amplitude distribution of clean speech can be approximated by
    the Gamma distribution with a shaping parameter of 0.4, and that an additive
    noise signal is Gaussian. Then we can estimate the SNR by examining the amplitude
    distribution of the noise-corrupted speech.

    References:
        [1] Chanwoo Kim, and Richard M. Stern. Robust Signal-to-Noise Ratio
            Estimation Based on Waveform Amplitude Distribution Analysis.
            in Proc. ISCA Interspeech, 2008, pp. 2598-2601.
        [2] https://labrosa.ee.columbia.edu/projects/snreval/#9

    Args:
        audio (np.ndarray): input audio signal (time, [channels])
        min_val (float): minimum value
    Returns:
        snr (list): estimated SNR (length = audio.shape[1])
    """
    audio = audio / np.max(np.abs(audio))
    audio = np.abs(audio)
    np.clip(audio, min_val, None, out=audio)
    # E[|z|]
    mean = np.mean(audio, axis=0)
    # E[log|z|]
    logmean = np.mean(np.log(audio), axis=0)
    # log(E[|z|]) - E[log(|z|)]
    diff = np.atleast_1d(np.log(mean) - logmean)

    dNoisyEng = np.sum(audio**2, axis=0)

    # Table interpolation
    snr = []
    # for ch in range(diff.shape[0]):
    #     idx = np.where(diff[ch] <= Gvals)[0]
    #     if len(idx) == 0:
    #         snr.append(dbvals[0])
    #     else:
    #         snr.append(dbvals[idx.min()])
    for ch in range(diff.shape[0]):
        # loop over each channel
        idx = np.where(Gvals < diff[ch])[0]
        if len(idx) == 0:
            snr_ = Gvals[0]
        else:
            snr_idx = idx.max() if len(idx) > 0 else 0
            if snr_idx >= len(Gvals):
                print("Warning: Cannot interpolate beyond the lookup table range")
                return None
            elif snr_idx == len(Gvals) - 1:
                snr_ = dbvals[snr_idx]
            else:
                # can actually interpolate
                snr_ = dbvals[snr_idx] + (diff[ch] - Gvals[snr_idx]) / (
                    Gvals[snr_idx + 1] - Gvals[snr_idx]
                ) * (dbvals[snr_idx + 1] - dbvals[snr_idx])
        snr.append(snr_)

    for i in range(len(snr)):
        dFactor = 10 ** (snr[i] / 10.0)
        dNoiseEng = dNoisyEng / (1 + dFactor)
        dSigEng = dNoisyEng * dFactor / (1 + dFactor)
        snr[i] = 10 * np.log10(dSigEng / dNoiseEng)
    return np.mean(snr)


################################################################
# Main entry
################################################################
def main(args):
    data_pairs = []
    with open(args.inf_scp, "r") as f:
        for line in f:
            uid, audio_path = line.strip().split()
            data_pairs.append((uid, audio_path))

    ret = process_map(
        process_one_pair,
        data_pairs,
        max_workers=args.nj,
        chunksize=args.chunksize,
    )

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    writers = {metric: (outdir / f"{metric}.scp").open("w") for metric in METRICS}

    for uid, score in ret:
        for metric, value in score.items():
            writers[metric].write(f"{uid} {value}\n")

    for metric in METRICS:
        writers[metric].close()

    with (outdir / "RESULTS.txt").open("w") as f:
        for metric in METRICS:
            mean_score = np.nanmean([score[metric] for uid, score in ret])
            f.write(f"{metric}: {mean_score:.4f}\n")
    print(f"Overall results have been written in {outdir / 'RESULTS.txt'}", flush=True)


def process_one_pair(data_pair):
    uid, inf_path = data_pair
    audio, fs = sf.read(inf_path, dtype="float32")

    scores = {}
    for metric in METRICS:
        if metric == "WADASNR":
            scores[metric] = wada_snr(audio)
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
        "--nj",
        type=int,
        default=8,
        help="Number of parallel workers to speed up evaluation",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=1000,
        help="Chunk size used in process_map",
    )
    args = parser.parse_args()

    main(args)
