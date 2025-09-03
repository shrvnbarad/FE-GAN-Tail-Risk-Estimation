# Risk Management with Feature-Enriched GANs (FE-GAN)

This repository contains the implementation of the research paper:  
**"Risk Management with Feature-Enriched Generative Adversarial Networks (FE-GAN)" by Ling Chen (UCL, 2024)**  
[Paper Link](https://arxiv.org/abs/2411.15519)

---

## 📌 Overview
Financial risk management relies heavily on accurate estimation of **Value at Risk (VaR)** and **Expected Shortfall (ES)**.  
This project implements **Feature-Enriched GANs (FE-GAN)**, an extension of standard GAN architectures that incorporates **additional input sequences from historical, GBM, and time-series models** to improve performance on temporal financial datasets.

We further implement and compare two specialized GANs:
- **Wasserstein GAN (WGAN)**
- **Tail-GAN (Tail-GAN)**

The FE-GAN framework significantly reduces estimation errors, accelerates convergence, and demonstrates superior accuracy in modeling financial risk measures.

---

## 🏗️ Architecture
- **Generator**: Extended with enriched input sequences (historical data, GBM, time-series, or hybrid).  
- **Discriminator**: Standard WGAN/Tail-GAN structure.  
- **Training**:
  - Batch size: 100  
  - Window size: 250 trading days  
  - Generator: 10 layers × 1000 units  
  - Discriminator: 5 layers × 100 units  

---

## 📊 Input Sequences
1. **Historical Data** – Past 250 data points.  
2. **GBM Model** – Mean and variance under Geometric Brownian Motion assumption.  
3. **Time-Series Models** – AR, MA, and ARMA with AIC/BIC-based selection.  
4. **Hybrid (Time-Series + GBM)** – Combines trend/seasonality from ARMA with volatility from GBM.  

---

## 🚀 Key Results
- **VaR Estimation**:
  - Historical and GBM-based inputs yield the strongest performance.  
  - Hybrid model improves accuracy by combining trend + volatility.  
- **ES Estimation**:
  - Tail-GAN consistently outperforms WGAN due to its loss function designed for tail risks.  
- **Training Efficiency**:
  - FE-GAN reduces training time by nearly **90%** compared to baseline GANs.  

---

## 📂 Repository Structure
```bash
├── GAN.ipynb               # Notebook
└── README.md
````

---

## ⚙️ Installation

```bash
git clone [https://github.com/yourusername/FE-GAN-RiskManagement.git](https://github.com/shrvnbarad/FE-GAN-Tail-Risk-Estimation.git)
```
---

## 📈 Example Results

* **VaR differences** reduced by up to **3×** compared to baseline.
* **ES estimation errors** consistently lower with FE-GAN, especially using Tail-GAN.

---

## 🔮 Future Work

* Incorporate **LSTM/Attention-based inputs** for better temporal modeling.
* Extend beyond VaR & ES to **stress testing** and **scenario generation**.
* Explore **dynamic input generation** (variable-length windows).

---

## 📜 Reference

Chen, Ling. *Risk Management with Feature-Enriched Generative Adversarial Networks (FE-GAN)*,
University College London, 2024. [arXiv:2411.15519](https://arxiv.org/abs/2411.15519)

---

## 🧑‍💻 Author

Implementation by **Shravan Pradeep Barad**
Based on the original research by **Ling Chen, UCL**.
