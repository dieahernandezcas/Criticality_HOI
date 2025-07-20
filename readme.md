# Brain State Modeling and Synthetic Data Generation with Wilson-Cowan and Neurolib

Source code for simulating different brain states using the Wilson-Cowan population model and generating synthetic neural data, followed by analyses including avalanche detection and higher-order interactions.

![Status Badge](https://img.shields.io/badge/Status-In%20Development-yellow) ![License Badge](https://img.shields.io/badge/License-MIT-blue) ![Version Badge](https://img.shields.io/badge/Version-1.0.0-informational)

---

## Project Overview

This project focuses on applying the **Wilson-Cowan (WC) model**, implemented in the `neurolib` library, to simulate neural activity across a whole-brain network. The main goal is to generate **synthetic brain signals** representing various neurological states (e.g., healthy, coma, brain death) by adjusting WC model parameters. These simulations are performed on an **80-channel structural connectivity matrix** derived from the Human Connectome Project (HCP) dataset.

This project aims to:

1. Simulate the dynamics of excitatory and inhibitory neural populations in both single-node and multi-node (80-channel) configurations.
2. Explore how varying key WC model parameters (e.g., external input, global coupling) can lead to different brain activity regimes.
3. Generate and store simulated data (continuous and binarized neural activity) for a cohort of 200 "patients," representing distinct clinical conditions.
4. Perform advanced analyses on the simulated data, including **brain avalanche detection** and calculation of **Higher-Order Interactions (HOI) via Cumulants**.
5. Provide a synthetic dataset and analysis tools useful for studying complex brain dynamics and developing new methods for neural data analysis.

---

## Folder Structure

The repository is organized as follows:

```
Repo_Root/
├── data/                     # Input and generated data files
│   ├── hcp/                  # HCP dataset (Cmat and Dmat, loaded by neurolib)
│   └── simulated/            # Simulated patient data (e.g., patient_001_healthy_critical_raw.pkl)
├── scripts/                  # Python scripts for simulations and analysis
│   ├── 01_WilsonCowan_1Chanels.py                  # Single-channel WC model regime exploration
│   ├── 02_WilsonCowan_80Chanels.py                 # 80-channel WC model exploration (e.g., specific conditions)
│   ├── 03_WilsonCowan_80Chanels_patients.py        # Data generation for multiple patients (200) across conditions
│   ├── 04_WilsonCowan_80Chanels_Criticality.ipynb  # Exploration of 80-channel WC model for criticality
│   ├── 05_Criticality_Avalanches.ipynb             # Brain avalanche detection and analysis from binarized data
│   └── 06_Avalanches_Cumulants.ipynb               # Higher-Order Interactions (HOI) calculation via cumulants
├── results/                  # Analysis outputs (e.g., avalanche stats, HOI results, plots)
├── .gitignore                # Ignored files (e.g., virtual envs, large data files)
└── README.md                 # Project documentation
```

---

## Setup and Installation

1. **Clone the repository:**
    ```bash
    git clone [YOUR_REPOSITORY_URL]
    cd [your_repository_name]
    ```

2. **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3. **Install dependencies:**
    The project requires `jupyter`, `neurolib`, `numpy`, `matplotlib`, and `pathlib`. Additional libraries might be required for specific analysis notebooks (e.g., `scipy` for cumulants).  
    ```bash
    pip install -r requirements.txt
    ```
    (If `neurolib` is not in your `requirements.txt`, install it manually: `pip install neurolib`)

---

## Usage

The `notebooks/` directory contains the main Jupyter notebooks. It's recommended to run them sequentially for a complete analysis pipeline.

1.  **Explore Single-Channel Wilson-Cowan Dynamics:**
    Open `01_WilsonCowan_1Chanels.ipynb` in Jupyter Notebook or Jupyter Lab.
    This notebook explores the fundamental behavior of a single Wilson-Cowan (WC) neural node under different external input regimes (subcritical, critical, hypercritical) and generates visualizations. It's a foundational step to understand the model's dynamics.

2.  **Explore 80-Channel Wilson-Cowan Dynamics (Individual Conditions):**
    Open `02_WilsonCowan_80Chanels.ipynb` in Jupyter Notebook or Jupyter Lab.
    This notebook allows you to simulate the 80-channel WC network for specific neurological conditions or parameter sets, providing insights into network-level behavior.

3.  **Generate Patient Data:**
    Open `03_WilsonCowan_80Chanels_patients.ipynb` in Jupyter Notebook or Jupyter Lab.
    This notebook is crucial for running the full simulation pipeline to generate raw EEG-like data for multiple synthetic patients (e.g., 200) across various conditions. The generated `.pkl` files (e.g., `patient_001_healthy_critical_raw.pkl`) will be saved in the `data/simulated/` folder.

4.  **Simulate and Explore 80-Channel Wilson-Cowan Dynamics (Criticality Focus for Avalanches):**
    Open `04_WilsonCowan_80Chanels_Criticality.ipynb` in Jupyter Notebook or Jupyter Lab.
    Building upon the previous steps, this notebook focuses specifically on simulating the 80-channel Wilson-Cowan network to achieve and explore **critical dynamics**, producing the binarized activity necessary for avalanche detection.

5.  **Detect Brain Avalanches:**
    After generating the binarized data in the previous step, proceed to `05_Criticality_Avalanches.ipynb`.
    This notebook is dedicated to **detecting brain avalanches** from the binarized activity. It analyzes their properties (e.g., size, duration distributions) and saves the segmented avalanche data (`binarized_during_avalanches` and `binarized_outside_avalanches` for each patient) in the `results/avalanches/` folder.

6.  **Calculate Higher-Order Interactions (HOI) via Cumulants:**
    Finally, open `06_Avalanches_Cumulants.ipynb`.
    This notebook computes **higher-order interactions (HOI)** using cumulants for the binarized activity. It's designed to compare these interactions between periods **during and outside avalanches**. The calculated cumulants for each patient are saved as `.pkl` files in the `results/cumulants/` folder.

---

## Contributing

Contributions are welcome. If you wish to contribute, please open an issue or submit a pull request.

---

## References

- **Neurolib:**  
  Official documentation of the `neurolib` + Wilson-Cowan library:  
  [https://neurolib-dev.github.io/examples/example-0.4-wc-minimal/](https://neurolib-dev.github.io/examples/example-0.4-wc-minimal/)

- **Wilson-Cowan Model:**  
  De Candia, Antonio, et al. *Critical Behaviour of the Stochastic Wilson-Cowan Model.*  
  *PLOS Computational Biology,* 17(8), e1008884, 2021.  
  [https://doi.org/10.1371/journal.pcbi.1008884](https://doi.org/10.1371/journal.pcbi.1008884)

- **Wilson-Cowan Model and Criticality:**  
  Alvankar Golpayegan, Hanieh, and Antonio De Candia. *Bistability and Criticality in the Stochastic Wilson-Cowan Model.*  
  *Physical Review E,* 107(3), 034404, 2023.  
  [https://doi.org/10.1103/PhysRevE.107.034404](https://doi.org/10.1103/PhysRevE.107.034404)

- **Brain Criticality / Avalanches:**  
  Larremore, Daniel B., et al. *Statistical Properties of Avalanches in Networks.*  
  *Physical Review E,* 85(6), 066131, 2012.  
  [https://doi.org/10.1103/PhysRevE.85.066131](https://doi.org/10.1103/PhysRevE.85.066131)

- **Higher-Order Interactions:**  
  Hindriks, Rikkert, et al. *Higher‐order Functional Connectivity Analysis of Resting‐state Functional Magnetic Resonance Imaging Data Using Multivariate Cumulants.*  
  *Human Brain Mapping,* 45(5), e26663, 2024.  
  [https://doi.org/10.1002/hbm.26663](https://doi.org/10.1002/hbm.26663)

---

## Contact

If you have questions, comments, or suggestions about this project, please contact:

- **Diego Alejandro Hernández Castañeda**  
  PhD Student, Systems and Computing Engineering  
  Universidad Nacional de Colombia  
  Email: dieahernandezcas@unal.edu.co  

- **Prof. Francisco Albeiro Gómez Jaramillo, PhD**  
  Associate Professor, Mathematics Department  
  Universidad Nacional de Colombia  
  Email: fagomezj@unal.edu.co  

- **Prof. Prejaas Kavish Baldewpersad Tewarie, PhD**  
  Clinical Physician and Teaching Assistant Grant Holder  
  CERVO Brain Research Centre, Université Laval  
  Email: Prejaas.KBTewarie@cervo.ulaval.ca  