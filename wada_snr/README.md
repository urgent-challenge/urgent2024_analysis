## Introduction

This folder contains the implementations of [WADA-SNR](calculate_wada_snr.py) algorithm.

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

    ```bash
    #!/bin/bash
    nj=16  # Number of parallel CPU jobs for speedup
    python=python3

    inf_scp=enhanced.scp
    output_prefix=outdir

    # Run parallel jobs on multiple CPUs
    ${python} calculate_wada_snr.py \
        --inf_scp "${inf_scp}" \
        --output_dir "${output_prefix}"/wada_snr \
        --nj ${nj} \
        --chunksize 100

The above command will run the WADA-SNR evaluation on the enhanced speech samples listed in `enhanced.scp` and save the results in the `outdir/wada_snr` directory.

The results will be saved in a scp file named `WADASNR.scp` and a text file named `RESULTS.txt`.
The `WADASNR.scp` file will contain the WADA-SNR scores for each enhanced speech sample, while the `RESULTS.txt` file will contain the average WADA-SNR score across all samples.
