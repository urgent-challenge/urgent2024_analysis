## Introduction

This folder contains the implementations of additional objective evaluation metrics for analyzing the results of the [URGENT Challenge](https://urgent-challenge.github.io/urgent2024/). The metrics include:

- [DNSMOS Pro](calculate_nonintrusive_dnsmos_pro.py)
- [UTMOS](calculate_nonintrusive_mos.py)
- [WV-MOS](calculate_nonintrusive_mos.py)
- [SCOREQ](calculate_nonintrusive_scoreq.py)
- [VQScore](calculate_nonintrusive_vqscore.py)

## Usage

1. Make sure that you have prepared the enhanced speech files under the same folder. For example, the enhanced speech samples are stored in `enhanced/` folder. The folder structure will look like this:
    ```
    📁 /path/to/your/data/
    └── 📁 enhanced/
        ├── 🔈 fileid_1.flac
        ├── 🔈 fileid_2.flac
        └── ...
    ```

2. Prepare the scp file for the enhanced speech signals. Each row in the scp files should contain two columns: the unique utterance ID and the corresponding path to the audio file. The scp files should look like this:
    ```
    # enhanced.scp
    fileid_1 /path/to/your/data/enhanced/fileid_1.flac
    fileid_2 /path/to/your/data/enhanced/fileid_2.flac
    ...
    ```

    You can prepare the scp files using the following command:
    ```bash
    find /path/to/your/data/enhanced/ -name "*.flac" | \
        awk -F'/' '{print $NF" "$0}' | sed 's/.flac / /' | \
        LC_ALL=C sort -u > enhanced.scp
    ```

3. Run the evaluation script with the following command:

    <details><summary>DNSMOS Pro (click to expand)</summary><div>

    ```bash
    #!/bin/bash
    nj=8  # Number of parallel CPU/GPU jobs for speedup
    python=python3

    git clone https://github.com/fcumlin/DNSMOSPro
    DNSMOSPro_dir="./DNSMOSPro"

    # Whether to use GPU for inference
    gpu_inference=true
    if ${gpu_inference}; then
        _device="cuda"
    else
        _device="cpu"
    fi

    inf_scp=enhanced.scp
    output_prefix=outdir

    pids=() # initialize pids
    for idx in $(seq ${nj}); do
    (

        # Run each parallel job on a different GPU (if $gpu_inference = true)
        CUDA_VISIBLE_DEVICES=$((${idx} - 1)) ${python} calculate_nonintrusive_dnsmos_pro.py \
            --inf_scp "${inf_scp}" \
            --output_dir "${output_prefix}"/scoring_dnsmos_pro \
            --device ${_device} \
            --nsplits ${nj} \
            --job ${idx} \
            --model_path "${DNSMOSPro_dir}"/runs/NISQA/model_best.pt

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs were failed." && false
    echo "Finished"

    if [ ${nj} -gt 1 ]; then
        for i in $(seq ${nj}); do
            cat "${output_prefix}"/scoring_dnsmos_pro/DNSMOSPro.${i}.scp
        done > "${output_prefix}"/scoring_dnsmos_pro/DNSMOSPro.scp

        python - <<EOF
    scores = []
    with open("${output_prefix}/scoring_dnsmos_pro/DNSMOSPro.scp", "r") as f:
        for line in f:
            uid, score = line.strip().split()
            scores.append(float(score))
    mean_score = np.nanmean(scores)
    with open("${output_prefix}/scoring_dnsmos_pro/RESULTS.txt", "w") as out:
        out.write(f"DNSMOSPro: {mean_score:.4f}\n")
    EOF
    fi
    ```

    </div></details>

    <details><summary>UTMOS & WV-MOS (click to expand)</summary><div>

    ```bash
    #!/bin/bash
    nj=8  # Number of parallel CPU/GPU jobs for speedup
    python=python3

    ${python} -m pip install git+https://github.com/AndreevP/wvmos
    ${python} -m pip install git+https://github.com/sarulab-speech/UTMOSv2.git
    git clone https://huggingface.co/spaces/sarulab-speech/UTMOSv2

    # Whether to use GPU for inference
    gpu_inference=true
    if ${gpu_inference}; then
        _device="cuda"
    else
        _device="cpu"
    fi

    inf_scp=enhanced.scp
    output_prefix=outdir

    pids=() # initialize pids
    for idx in $(seq ${nj}); do
    (

        # Run each parallel job on a different GPU (if $gpu_inference = true)
        CUDA_VISIBLE_DEVICES=$((${idx} - 1)) ${python} calculate_nonintrusive_mos.py \
            --inf_scp "${inf_scp}" \
            --output_dir "${output_prefix}"/scoring_nn_mos \
            --device ${_device} \
            --nsplits ${nj} \
            --job ${idx} \
            --utmos_tag utmos22_strong

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs were failed." && false
    echo "Finished"

    if [ ${nj} -gt 1 ]; then
        for i in $(seq ${nj}); do
            cat "${output_prefix}"/scoring_nn_mos/UTMOS.${i}.scp
        done > "${output_prefix}"/scoring_nn_mos/UTMOS.scp
        for i in $(seq ${nj}); do
            cat "${output_prefix}"/scoring_nn_mos/WV_MOS.${i}.scp
        done > "${output_prefix}"/scoring_nn_mos/WV_MOS.scp

        python - <<EOF
    scores = []
    with open("${output_prefix}/scoring_nn_mos/UTMOS.scp", "r") as f:
        for line in f:
            uid, score = line.strip().split()
            scores.append(float(score))
    mean_score = np.nanmean(scores)
    with open("${output_prefix}/scoring_nn_mos/RESULTS.txt", "w") as out:
        out.write(f"UTMOS: {mean_score:.4f}\n")

    scores = []
    with open("${output_prefix}/scoring_nn_mos/WV_MOS.scp", "r") as f:
        for line in f:
            uid, score = line.strip().split()
            scores.append(float(score))
    mean_score = np.nanmean(scores)
    with open("${output_prefix}/scoring_nn_mos/RESULTS.txt", "a") as out:
        out.write(f"WV_MOS: {mean_score:.4f}\n")
    EOF
    fi
    ```

    </div></details>

    <details><summary>SCOREQ (click to expand)</summary><div>

    ```bash
    #!/bin/bash
    nj=8  # Number of parallel CPU/GPU jobs for speedup
    python=python3

    ${python} -m pip install scoreq

    # Whether to use GPU for inference
    gpu_inference=true
    if ${gpu_inference}; then
        _device="cuda"
    else
        _device="cpu"
    fi

    inf_scp=enhanced.scp
    output_prefix=outdir

    pids=() # initialize pids
    for idx in $(seq ${nj}); do
    (

        # Run each parallel job on a different GPU (if $gpu_inference = true)
        CUDA_VISIBLE_DEVICES=$((${idx} - 1)) ${python} calculate_nonintrusive_scoreq.py \
            --inf_scp "${inf_scp}" \
            --output_dir "${output_prefix}"/scoring_scoreq \
            --device ${_device} \
            --nsplits ${nj} \
            --job ${idx}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs were failed." && false
    echo "Finished"

    if [ ${nj} -gt 1 ]; then
        for i in $(seq ${nj}); do
            cat "${output_prefix}"/scoring_scoreq/SCOREQ.${i}.scp
        done > "${output_prefix}"/scoring_scoreq/SCOREQ.scp

        python - <<EOF
    scores = []
    with open("${output_prefix}/scoring_scoreq/SCOREQ.scp", "r") as f:
        for line in f:
            uid, score = line.strip().split()
            scores.append(float(score))
    mean_score = np.nanmean(scores)
    with open("${output_prefix}/scoring_scoreq/RESULTS.txt", "w") as out:
        out.write(f"SCOREQ: {mean_score:.4f}\n")
    EOF
    fi
    ```

    </div></details>

    <details><summary>VQScore (click to expand)</summary><div>

    ```bash
    #!/bin/bash
    nj=8  # Number of parallel CPU/GPU jobs for speedup
    python=python3

    git clone https://github.com/JasonSWFu/VQscore
    VQscore_dir=VQscore

    # Whether to use GPU for inference
    gpu_inference=true
    if ${gpu_inference}; then
        _device="cuda"
    else
        _device="cpu"
    fi

    inf_scp=enhanced.scp
    output_prefix=outdir

    pids=() # initialize pids
    for idx in $(seq ${nj}); do
    (

        # Run each parallel job on a different GPU (if $gpu_inference = true)
        CUDA_VISIBLE_DEVICES=$((${idx} - 1)) ${python} calculate_nonintrusive_vqscore.py \
            --inf_scp "${inf_scp}" \
            --output_dir "${output_prefix}"/scoring_vqscore \
            --device ${_device} \
            --nsplits ${nj} \
            --job ${idx} \
            --vqscore_conf "${VQscore_dir}/config/QE_cbook_size_2048_1_32_IN_input_encoder_z_Librispeech_clean_github.yaml" \
            --vqscore_model "${VQscore_dir}/exp/QE_cbook_size_2048_1_32_IN_input_encoder_z_Librispeech_clean_github/checkpoint-dnsmos_ovr_CC=0.835.pkl"

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs were failed." && false
    echo "Finished"

    if [ ${nj} -gt 1 ]; then
        for i in $(seq ${nj}); do
            cat "${output_prefix}"/scoring_vqscore/VQscore.${i}.scp
        done > "${output_prefix}"/scoring_vqscore/VQscore.scp

        python - <<EOF
    scores = []
    with open("${output_prefix}/scoring_vqscore/VQscore.scp", "r") as f:
        for line in f:
            uid, score = line.strip().split()
            scores.append(float(score))
    mean_score = np.nanmean(scores)
    with open("${output_prefix}/scoring_vqscore/RESULTS.txt", "w") as out:
        out.write(f"VQscore: {mean_score:.4f}\n")
    EOF
    fi
    ```

    </div></details>

The above commands will run evaluation on the enhanced speech samples listed in `enhanced.scp` and save the results in the `outdir/scoring_*` directory.

The results will be saved in a scp file named `*.scp` and a text file named `RESULTS.txt` under a subdirectory corresponding to each metric.
The scp file will contain the detailed metric value for each enhanced speech sample, while the `RESULTS.txt` file will contain the average metric value across all samples.
