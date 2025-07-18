# Brain State Modeling and Synthetic Data Generation with Wilson-Cowan and Neurolib

Source code for simulating different brain states using the Wilson-Cowan population model and generating synthetic neural data, followed by analyses including avalanche detection and higher-order interactions.

![Status Badge](https://img.shields.io/badge/Status-In%20Development-yellow) ![License Badge](https://img.shields.io/badge/License-MIT-blue) ![Version Badge](https://img.shields.io/badge/Version-1.0.0-informational)

---

## Project Overview

This project focuses on applying the **Wilson-Cowan (WC) model**, implemented in the `neurolib` library, to simulate neural activity across a whole-brain network. The main goal is to generate **synthetic brain signals** representing various neurological states (e.g., healthy, coma, brain death) by adjusting WC model parameters. These simulations are performed on an **80-channel structural connectivity matrix** derived from the Human Connectome Project (HCP) dataset.

This project aims to:

1.  Simulate the dynamics of excitatory and inhibitory neural populations in both single-node and multi-node (80-channel) configurations.
2.  Explore how varying key WC model parameters (e.g., external input, global coupling) can lead to different brain activity regimes.
3.  Generate and store simulated data (continuous and binarized neural activity) for a cohort of 200 "patients," representing distinct clinical conditions.
4.  Perform advanced analyses on the simulated data, including **brain avalanche detection** and calculation of **Higher-Order Interactions (HOI) via Cumulants**.
5.  Provide a synthetic dataset and analysis tools useful for studying complex brain dynamics and developing new methods for neural data analysis.

---

## Folder Structure

The repository is organized as follows:

.
Repo_Root/
├── data/              # All generated and input data files
│   ├── hcp/           # HCP dataset with Cmat and Dmat (loaded by neurolib)
│   └── simulated/     # Simulated patient data (e.g., patient_001_healthy_critical_raw.pkl)
├── scripts/           # Python scripts for simulations and analysis
│   ├── 01_WilsonCowan_1Chanels.py         # Script for single-channel WC model dynamic regime exploration.
│   ├── 02_WilsonCowan_80Chanels.py        # Script for 80-channel WC model exploration (e.g., specific conditions).
│   ├── 03_WilsonCowan_80Chanels_patients.py # Script for generating and saving data for multiple patients (200) across various conditions.
│   ├── 04_avalanches_detection.py         # Script for detecting and analyzing brain avalanches from binarized data.
│   └── 05_HOI_Cumulants.py                # Script for calculating Higher-Order Interactions (HOI) via cumulants.
├── results/           # Outputs from analyses (e.g., avalanche statistics, HOI results, plots).
├── .gitignore         # Specifies intentionally untracked files (e.g., virtual environments, large data files)
└── README.md          # This file


---

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [YOUR_REPOSITORY_URL]
    cd [your_repository_name]
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    The core project depends on `neurolib`, `numpy`, `matplotlib`, and `pathlib`. Additional libraries might be required for specific analysis scripts (e.g., `scipy` for cumulants). Ensure you have `neurolib` installed. You can create a `requirements.txt` file by running `pip freeze > requirements.txt` after installing the necessary libraries.
    ```bash
    pip install -r requirements.txt
    ```
    (If `neurolib` is not in your `requirements.txt`, install it manually: `pip install neurolib`)

---

## Usage

The `scripts/` directory contains the main executables for this project. It's recommended to run them in the order of their numbering for a full pipeline.

1.  **Explore Single-Channel Wilson-Cowan Dynamics:**
    * Run `01_WilsonCowan_1Chanels.py` to understand the fundamental behavior of a single WC node under different external input regimes (subcritical, critical, hypercritical). This script generates plots for visualization.
    ```bash
    python scripts/01_WilsonCowan_1Chanels.py
    ```

2.  **Explore 80-Channel Wilson-Cowan Dynamics (Individual Conditions):**
    * Run `02_WilsonCowan_80Chanels.py` to simulate the 80-channel network for specific, individual neurological conditions. This script also generates plots.
    ```bash
    python scripts/02_WilsonCowan_80Chanels.py
    ```

3.  **Generate Patient Data:**
    * Execute `03_WilsonCowan_80Chanels_patients.py` to perform the full simulation pipeline for 200 patients across various conditions. This script will generate and save the raw and binarized simulated data files.
    ```bash
    python scripts/03_WilsonCowan_80Chanels_patients.py
    ```
    The generated `.pkl` files (e.g., `patient_001_healthy_critical_raw.pkl`, `patient_001_healthy_critical_binarized.pkl`) will be stored in the `data/simulated/` directory.

4.  **Detect Brain Avalanches:**
    * After generating patient data, run `04_avalanches_detection.py`. This script will load the binarized data from `data/simulated/` and perform avalanche detection and analysis. Results (e.g., avalanche size/duration distributions) will be saved in `results/`.
    ```bash
    python scripts/04_avalanches_detection.py
    ```

5.  **Calculate Higher-Order Interactions (HOI) via Cumulants:**
    * Finally, run `05_HOI_Cumulants.py`. This script will load the appropriate simulated data (likely raw or binarized, depending on the HOI method) and compute higher-order interactions using cumulants. Results will be stored in `results/`.
    ```bash
    python scripts/05_HOI_Cumulants.py
    ```

---

## Contributing

Contributions are welcome. If you wish to contribute, please open an issue or submit a pull request.

---

## References

* **Wilson-Cowan Model:** Wilson, H. R., & Cowan, J. D. (1972). Excitatory and inhibitory interactions in localized populations of model neurons. *Biophysical Journal, 12*(1), 1-24.
* **Neurolib:** For more details on the `neurolib` library, refer to its [official documentation](https://neurolib-dev.net/docs/).
* **Brain Criticality / Avalanches:** Beggs, J. M., & Plenz, D. (2003). Neuronal avalanches in neocortical circuits. *Journal of Neuroscience, 23*(34), 11167-11177.
* **Higher-Order Interactions:** (Relevant papers on HOI and cumulants can be added here once specific methods are chosen).

---

## Contact

If you have questions, comments, or suggestions about this project, please contact:

Diego Alejandro Hernández Castañeda - dieahernandezcas@unal.edu.co
Francisco Albeiro Gomez Jaramillo - fagomezj@unal.edu.co
Prejaas Kavish Baldewpersad Tewarie - Prejaas.K.B.Tewarie@cervo.ulaval.ca
 