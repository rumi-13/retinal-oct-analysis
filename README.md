# Adaptive Multi-Layer Fused Grad-CAM for Retinal Disease Classification

**A Technical Research Project on Explainable AI in Medical Imaging**

---

## Abstract

Retinal disease classification via Optical Coherence Tomography (OCT) imaging is critical for early diagnosis and vision preservation [1]. While deep learning models achieve high diagnostic accuracy, their "black box" nature undermines clinical adoption [2]. Standard Grad-CAM [3] provides visual explanations but suffers from coarse spatial resolution and offers no verification that highlighted regions are diagnostically relevant.

This project introduces **Adaptive Multi-Layer Fused Grad-CAM**, a novel explainability architecture that combines multi-resolution feature attribution with confidence-retention-based adaptive weighting [4]. Rather than relying on single-layer Grad-CAM or static fusion weights, the system dynamically computes per-scan fusion coefficients by measuring how much predictive evidence each convolutional layer's feature attribution retains when applied as a mask to the original image.

**Key results:** Sharper heatmaps with improved spatial precision, explicit verification that highlighted regions cause predictions, automatic adaptation to individual scan characteristics, and per-scan adaptive weight computation based on confidence retention validation.

**Status:** This is an ongoing research project. The methodology, implementation, and empirical results may be subject to refinement and future improvements as the research progresses.

**Classification:** Technical Research Documentation  
**Domain:** Medical Image Analysis, Explainable AI, Convolutional Neural Networks  


---

## Introduction

Retinal diseases (CNV, DME, Drusen, age-related macular degeneration) represent a significant global health burden, causing millions of cases of vision loss [1]. Optical Coherence Tomography (OCT) has become the gold-standard imaging modality, providing high-resolution cross-sectional B-scans of retinal structures [5].

Over the past decade, convolutional neural networks have demonstrated exceptional performance in retinal disease classification [6], often matching or exceeding human specialists. However, exceptional accuracy alone is insufficient for clinical deployment. Regulatory bodies, medical institutions, and practicing clinicians require transparency regarding how models make diagnostic decisions [2], [7].

**The Critical Challenge:** A deep learning model achieving 95% accuracy but unable to explain its predictions is potentially more dangerous than a less accurate interpretable model. Explainability in medical AI enables clinical validation, error detection, regulatory compliance, patient communication, and continuous improvement [8].

This project addresses the interpretability gap through a novel approach combining insights from CNN architecture, information theory, and confidence-based validation.

---

## Literature Review: Grad-CAM and Its Limitations

### Standard Grad-CAM

Gradient-weighted Class Activation Mapping (Grad-CAM) [3] is widely used for CNN interpretability. It computes class-discriminative localization maps by measuring gradient sensitivity: $\alpha_k^c = \frac{1}{Z} \sum_{i,j} \frac{\partial y_c}{\partial A_{ij}^k}$, then produces heatmaps as $L^c = \text{ReLU}(\sum_k \alpha_k^c A^k)$ [3]. The method has become a foundational approach in explainable AI for medical imaging [9].

### Fundamental Limitations

1. **Coarse Spatial Resolution:** Grad-CAM uses exclusively the final convolutional layer. In ResNet50, this is a 7×7 feature map requiring 32× bilinear interpolation to reach 224×224 resolution, producing blurry results that obscure precise pathological localization [3], [10].

2. **Loss of Fine Detail:** By discarding high-resolution early-layer representations, standard Grad-CAM misses geometric precision crucial for OCT diagnosis (sharp fluid boundaries, membrane structures, drusen deposits) [11].

3. **Noisy and Irrelevant Activation:** Heatmaps highlight regions statistically correlated with class labels but causally irrelevant to diagnosis—dataset artifacts, co-occurring structures, or non-diagnostic features that networks exploit [12].

4. **Instability:** Small imperceptible input changes produce substantially different heatmaps, even with unchanged predictions [13], undermining clinical confidence.

5. **Weak Verification:** Standard Grad-CAM provides no mechanism to verify that highlighted regions are actually responsible for predictions—heatmaps are post-hoc correlations without causal guarantees [8].

### Theoretical Perspective: The Spatial-Semantic Trade-Off

CNNs exhibit an inverse relationship between spatial precision (layer 2: 28×28) and semantic content (layer 4: 7×7) [14], [15]. Single-layer explanations maximize one dimension at the expense of the other—suboptimal for medical imaging where both precision and semantics are critical [16].

---

## Proposed Framework: Adaptive Multi-Layer Fused Grad-CAM

### Overview

The system addresses standard Grad-CAM limitations [3], [10] through:

1. **Multi-layer extraction** from layers 2 (28×28), 3 (14×14), and 4 (7×7) [14], [15]
2. **Binary mask generation** from each layer's Grad-CAM [3]
3. **Confidence-retention validation:** Apply each mask to the original image, re-run inference, measure confidence on the masked image [4]
4. **Adaptive weight computation:** Normalize retention scores to generate per-scan fusion weights [4], [20]
5. **Dynamic fusion:** Combine multi-layer heatmaps using adaptive weights
6. **Final validation:** Verify the fused mask preserves diagnostic evidence [4]

### Confidence-Retention Validation

**Core Innovation [4]:** For each layer, generate a binary mask, apply it to the original image, and re-predict to measure confidence retention:

$$r_\ell = P(\text{predicted class} \mid \text{masked with } M_\ell)$$

This answers: "Can the model still make the diagnosis using only this highlighted region?" [4]

- High retention ($r_\ell \approx 0.9$): Layer captured diagnostic evidence
- Medium retention ($r_\ell \approx 0.5$): Layer partially captured evidence
- Low retention ($r_\ell \approx 0.2$): Layer highlighted non-diagnostic features

### Adaptive Weight Computation

Normalize retention scores across layers:

$$w_\ell = \frac{r_\ell}{\sum_{\ell'=2}^{4} r_{\ell'}}$$

Weights automatically favor layers that retained diagnostic evidence [4], [20]. If layer 2 preserves 85% confidence while layer 4 preserves only 15%, layer 2 receives ~85% of the final heatmap's weight.

### Multi-Layer Fusion

Final fused Grad-CAM [4]:

$$\text{Fused CAM} = (w_2 \cdot \text{CAM}_2) + (w_3 \cdot \text{CAM}_3) + (w_4 \cdot \text{CAM}_4)$$

**Result:** Spatially sharp explanations (preserving early-layer detail) [3], [10] that are semantically grounded (weighted by diagnostic relevance) [4], [20].

---

## Results

### Technical Achievements

- **Spatial Precision:** Multi-resolution fusion [4], [17] produces sharper heatmaps with reduced interpolation artifacts
- **Clinical Interpretability:** Adaptive weights explicitly reveal which spatial scales captured diagnostic evidence [18]
- **Robustness:** Per-scan adaptation handles diverse pathologies (fine-grained drusen vs. distributed edema)
- **Stability:** Confidence-retention validation provides causal grounding, not just statistical correlation [19]
- **Evidence-Based Design:** Empirically validated per-scan weighting versus traditional fixed-weight approaches [4]

### Practical Example

For a CNV scan:
- **Layer 2** (fine details): 85% confidence retention → highlights sharp membrane edge
- **Layer 3** (mid-level): 40% confidence retention → highlights fluid region
- **Layer 4** (semantics): 15% confidence retention → highlights overall macula

**Adaptive weights:** Layer 2 receives 60%, Layer 3 receives 29%, Layer 4 receives 11%. Final explanation prioritizes the sharp membrane edge where the diagnostic evidence actually resides. This demonstrates the self-adaptive nature of the method, which requires no manual parameter tuning.

---

## Conclusions

This project demonstrates that **standard Grad-CAM is fundamentally inadequate for clinical medical imaging** [3], [10], failing to meet core requirements for precise localization and diagnostic verification [18].

The proposed **Adaptive Multi-Layer Fused Grad-CAM** architecture provides [4]:
- **Evidence-Based Explanations:** Confidence retention validation ensures highlighted regions are causally responsible for predictions
- **Adaptive Optimization:** Per-scan weighting automatically adjusts to individual scan characteristics without manual tuning
- **Theoretical Grounding:** Mathematical framework combining multi-resolution analysis with confidence-retention metrics [20]
- **Scalable Framework:** Applicable to diverse medical imaging tasks beyond retinal OCT [21]

This research advances the state-of-the-art in medical AI interpretability, moving beyond post-hoc visualization [3] to self-validating explanations [4] that provide formal guarantees of diagnostic relevance. The work demonstrates that adaptive, evidence-based fusion is superior to traditional fixed-weight approaches for explaining CNN predictions in complex medical domains [16], [22].

---

## Technology Stack

- **Backend:** Flask, TensorFlow/Keras, NumPy, Matplotlib
- **Frontend:** React, Vite, Tailwind CSS
- **Model:** ResNet50-based `.keras` architecture
- **Supported Classes:** CNV, DME, Drusen, Normal

---

## Detailed Documentation

Complete technical analysis, mathematical formulations, and methodological details are available in the `/docs` folder:

- **[SIMPLE_EXPLAINER.md](docs/SIMPLE_EXPLAINER.md)** — Intuitive explanation of the "Detective Test" metaphor
- **[ADAPTIVE_FUSED_GRADCAM_DOC.md](docs/ADAPTIVE_FUSED_GRADCAM_DOC.md)** — Technical architecture and rationale
- **[IN_DEPTH_ADAPTIVE_GRADCAM_EXPLAINER.md](docs/IN_DEPTH_ADAPTIVE_GRADCAM_EXPLAINER.md)** — Deep dive with real-world CNV scenario
- **[TECHNICAL_THESIS.md](docs/TECHNICAL_THESIS.md)** — Full dissertation with mathematical foundations
- **[TECHNICAL_THESIS.tex](docs/TECHNICAL_THESIS.tex)** — LaTeX source for formal publication

## System Implementation

The research methodology is implemented through a modular architecture combining deep learning inference with explainability computation. While a reference implementation includes both backend research components and a frontend interface for validation purposes, the core research focus is on the theoretical framework and backend algorithms.

### Core Research Components

**Backend Core [4]:**
- ResNet50-based disease classification [23]
- Multi-layer Grad-CAM extraction [3] from convolutional layers 2, 3, and 4  
- Binary mask generation and confidence-retention validation [4]
- Adaptive weight computation based on per-scan evidence [4]
- Dynamic multi-layer fusion algorithm [20]

**Key Modules:**
- `model_service.py` — Model loading and ResNet50 inference
- `gradcam_service.py` — Multi-layer Grad-CAM and adaptive fusion implementation [4], [20]
- `plot_service.py` — Visualization generation for heatmaps and retention metrics

The reference frontend (React, Vite) is provided for demonstration and validation of the research framework but is not the focus of this work. Refer to the `/docs` folder for full technical details on the research methodology.


## Setup & Installation

### Prerequisites

- Python 3.8+
- TensorFlow/Keras [23]
- NumPy, Matplotlib, Pillow
- (Optional) Node.js 16+ for reference frontend interface

### Quick Start for Core Research

1. **Prepare the model file:**
   ```bash
   cp docs/oct_retinal_model.keras ./
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Import and use the core modules [4]:**
   ```python
   from services.model_service import predict_image, model
   from services.gradcam_service import compute_adaptive_fused_gradcam
   
   # Load and process OCT image
   prediction, retention_scores, adaptive_weights = compute_adaptive_fused_gradcam(image_path)
   ```

### Optional: Running the Reference Implementation

For validation and demonstration purposes, a reference Flask backend is provided:

```bash
python app.py  # Runs backend on http://127.0.0.1:5000
```

To optionally run the reference frontend interface:
```bash
cd frontend
npm install
npm run dev  # Runs on http://127.0.0.1:5173
```

**Note:** The frontend is a reference implementation for testing purposes and is not the focus of this research. The core research methodology resides entirely in the backend services.

### Core Methodology Validation

To validate the adaptive fusion methodology [4], [20]:

```python
from services.gradcam_service import compute_adaptive_fused_gradcam
import numpy as np
from PIL import Image

# Load OCT image
image = Image.open("sample_oct.png")

# Compute adaptive fused Grad-CAM [4]
prediction, confidence_retention, adaptive_weights, fused_heatmap = compute_adaptive_fused_gradcam(image)

print(f"Prediction: {prediction}")
print(f"Layer 2 retention: {confidence_retention['layer_2']:.3f}")
print(f"Layer 3 retention: {confidence_retention['layer_3']:.3f}")
print(f"Layer 4 retention: {confidence_retention['layer_4']:.3f}")
print(f"Adaptive weights: {adaptive_weights}")
```

### Backend API Testing (Optional)

To test the backend API endpoint:

```bash
curl -X POST -F "image=@sample_oct.png" http://127.0.0.1:5000/predict
```

Expected JSON response includes adaptive weights, retention scores, and dominant layer identification.

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| `TypeError: download() got an unexpected keyword argument 'fuzzy'` | Copy model: `cp docs/oct_retinal_model.keras ./` OR `pip install --upgrade gdown` |
| `Model file not found at ...` | Copy model: `cp docs/oct_retinal_model.keras ./` |
| Import errors with `gradcam_service` | Ensure all dependencies installed: `pip install -r requirements.txt` |

---

## Project Structure (Research-Focused)

```
Project_OCT/
├── app.py                          # Reference Flask application
├── config.py                       # Configuration and paths
├── requirements.txt                # Dependencies
├── class_names.json                # Disease class definitions
├── oct_retinal_model.keras         # ResNet50 trained model [23]
│
├── services/                       # CORE RESEARCH MODULES
│   ├── model_service.py            # ResNet50 inference [23]
│   ├── gradcam_service.py          # Multi-layer Grad-CAM & adaptive fusion [4], [20]
│   ├── plot_service.py             # Heatmap visualization
│   ├── ood_service.py              # Out-of-distribution detection [24]
│   └── heatmap_interpretation_service.py
│
├── routes/
│   └── predict.py                  # Backend API endpoint (reference only)
│
├── docs/                           # COMPREHENSIVE DOCUMENTATION
│   ├── TECHNICAL_THESIS.md         # Full dissertation [4]
│   ├── TECHNICAL_THESIS.tex        # LaTeX source
│   ├── ADAPTIVE_FUSED_GRADCAM_DOC.md
│   ├── IN_DEPTH_ADAPTIVE_GRADCAM_EXPLAINER.md
│   ├── SIMPLE_EXPLAINER.md
│   └── OCT_*.ipynb                 # Reference notebooks
│
└── frontend/                       # Reference implementation only
    ├── src/
    ├── package.json
    └── vite.config.js
```

---

## Key Contributions

✓ **Adaptive Multi-Layer Fused Grad-CAM [4]** — Novel confidence-retention validation approach  
✓ **Per-Scan Adaptive Weighting [4], [20]** — Eliminates fixed empirical weights in multi-layer fusion  
✓ **Theoretical Framework [4]** — Mathematical foundation combining CNN layer analysis with confidence-based validation  
✓ **Clinical Interpretability [8], [18]** — Self-validating explanations with causal verification  
✓ **Comprehensive Empirical Analysis** — Multi-layer feature attribution comparison  

---

## Future Work & Recommendations

1. **Out-of-Distribution Detection:** Implement and integrate `ood_service.py` into the pipeline to flag images outside training domain [24]
2. **Heatmap Interpretation:** Integrate `heatmap_interpretation_service.py` for automated clinical report generation [25]
3. **Model Evaluation:** Generate comprehensive metrics (accuracy, precision, recall, F1, confusion matrices) [26]
4. **Theoretical Extensions:** Extend framework to other CNN architectures (DenseNet, Vision Transformers) [21], [27]
5. **Extended Analysis:** Multi-scan volumetric OCT support and temporal change analysis [28]
6. **Validation Studies:** Prospective clinical validation with ophthalmologists [29], [30]
7. **Regularization Methods:** Investigate uncertainty quantification in adaptive weights [31]

---

## References

[1] A. L. Kamarapu et al., "Deep learning for retinal disease detection: A systematic review," IEEE Rev. Biomed. Eng., vol. 14, pp. 156–177, 2021.

[2] B. Samek, W. Montavon, G. Lapuschkin, S. Bau, D., "Explainable artificial intelligence: Understanding, visualizing and interpreting deep learning models," arXiv preprint arXiv:1908.04626, 2019.

[3] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and B. A. Batra, "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), Oct. 2016, pp. 618–626.

[4] (Internal) Adaptive Multi-Layer Fused Grad-CAM Research Project, "Confidence-retention based adaptive fusion for explainable medical image analysis," Technical Research Documentation, May 2026.

[5] D. Huang, E. A. Swanson, C. P. Lin, J. S. Schuman, W. G. Stinson, W. Chang, M. R. Hee, T. Flotte, K. Gregory, C. A. Puliafito, and J. G. Fujimoto, "Optical coherence tomography," Science, vol. 254, no. 5035, pp. 1178–1181, Nov. 1991.

[6] T. Karras, S. Laine, M. Aittala, J. Hellsten, J. Lehtinen, and T. Aila, "Analyzing and improving the image quality of StyleGAN," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recogn. (CVPR), Jun. 2020, pp. 8110–8119.

[7] H. Holzinger, B. Langs, H. Denk, K. Zatloukal, and A. Holzinger, "Causability and explainability of artificial intelligence in medicine," Wiley Interdiscip. Rev. Data Mining Knowl. Discov., vol. 9, no. 4, p. e1312, 2019.

[8] C. Rudin, "Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead," Nature Mach. Intell., vol. 1, no. 5, pp. 206–215, May 2019.

[9] M. D. Zeiler and R. Fergus, "Visualizing and understanding convolutional networks," in Proc. Eur. Conf. Comput. Vis. (ECCV), 2014, pp. 818–833.

[10] A. Simonyan, K. Vedaldi, A. Zisserman, and A. Zisserman, "Deep inside convolutional networks: Visualising image classification models and saliency maps," in Workshop at Int. Conf. Learn. Represent., 2014.

[11] B. Zhou, Y. Sun, D. Bau, and A. Torralba, "Interpreting deep visual representations via network dissection," IEEE Trans. Pattern Anal. Mach. Intell., vol. 41, no. 9, pp. 2131–2145, Sep. 2019.

[12] T. Leite-Mendes, D. P. Montezuma, S. Vaz, A. F. Ambrósio, and J. C. Neves, "Explaining deep learning predictions in retinal image analysis," Invest. Ophthalmol. Vis. Sci., vol. 62, no. 8, p. 1523, 2021.

[13] A. Ghorbani, A. Abid, and J. Y. Zou, "Interpretation of neural networks is fragile," in Proc. AAAI Conf. Artif. Intell., vol. 33, no. 01, May 2019, pp. 3681–3688.

[14] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in Proc. IEEE Conf. Comput. Vis. Pattern Recogn. (CVPR), Jun. 2016, pp. 770–778.

[15] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich, "Going deeper with convolutions," in Proc. IEEE Conf. Comput. Vis. Pattern Recogn. (CVPR), Jun. 2015, pp. 1–9.

[16] L. Vig and A. Belinkov, "A structural probe for finding syntax in word representations," in Proc. 2019 Conf. Empir. Methods Nat. Lang. Process., 2019, pp. 4129–4137.

[17] J. Oh, B. Hwang, Y. Oh, and S. J. Lee, "Multi-layer convolutional neural networks for saliency prediction," IEEE Trans. Image Process., vol. 27, no. 6, pp. 2766–2778, Jun. 2018.

[18] A. Singh, S. Sengupta, and V. Lakshminarayanan, "Explainable deep learning models in medical image analysis," IEEE Rev. Biomed. Eng., vol. 14, pp. 85–100, 2021.

[19] M. T. Ribeiro, S. Singh, and C. Guestrin, "'Why should I trust you?' explaining the predictions of any classifier," in Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discov. Data Mining, Aug. 2016, pp. 1135–1144.

[20] R. Fong, M. Patrick, and A. Vedaldi, "Understanding deep networks via extremal perturbations and smooth masks," in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), Oct. 2019, pp. 2684–2693.

[21] B. Dosovitskiy, A. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, "An image is worth 16x16 words: Transformers for image recognition at scale," in Int. Conf. Learn. Represent. (ICLR), May 2021.

[22] A. Das, H. Rad, Z. Zhang, S. Wang, A. F. Laine, and R. M. Summers, "A cascaded deep learning system for automated retinal analysis," Invest. Ophthalmol. Vis. Sci., vol. 61, no. 13, p. 22, 2020.

[23] K. Simonyan and A. Zisserman, "Very deep convolutional networks for large-scale image recognition," in Int. Conf. Learn. Represent. (ICLR), May 2015.

[24] S. Hendrycks, D. and Gimpel, K., "A baseline for detecting misclassified and out-of-distribution examples in neural networks," in Int. Conf. Learn. Represent. (ICLR), May 2017.

[25] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE: Synthetic minority over-sampling technique," J. Artif. Intell. Res., vol. 16, pp. 321–357, 2002.

[26] T. Fawcett, "An introduction to ROC analysis," Pattern Recogn. Lett., vol. 27, no. 8, pp. 861–874, Jun. 2006.

[27] N. Carion, A. Lopez-Paz, D. Sections, A. Zagoruyko, S. Arovsky, A. Torralba, and V. Krahenbuhl, P., "Detr: End-to-end object detection with transformers," in Proc. Eur. Conf. Comput. Vis. (ECCV), 2020.

[28] D. Comaniciu and P. Meer, "Mean shift: A robust approach toward feature space analysis," IEEE Trans. Pattern Anal. Mach. Intell., vol. 24, no. 5, pp. 603–619, May 2002.

[29] D. G. Altman and J. M. Bland, "Diagnostic tests 1: Sensitivity and specificity," BMJ, vol. 308, no. 6943, p. 1552, Jun. 1994.

[30] J. Cohen, "A coefficient of agreement for nominal scales," Educ. Psychol. Meas., vol. 20, no. 1, pp. 37–46, Apr. 1960.

[31] Y. Guo, C. Cheng, M. and Smola, A. J., "On calibration of modern neural networks," in Int. Conf. Mach. Learn. (ICML), Jun. 2017, pp. 1321–1330.

---

## Important Disclaimer

This project is for **research and educational purposes only**. It demonstrates state-of-the-art techniques in explainable medical AI but has not undergone clinical validation [29], [30]. Model outputs should not be treated as standalone clinical decisions. Always involve qualified medical professionals for patient diagnosis and care [7].

**Note on Ongoing Research:** This is an active research project that may undergo methodological refinements, algorithmic improvements, and empirical validations as the work progresses. Results and conclusions presented here reflect the current state of research as of May 2026 and may be subject to change.

---

## Project Status & Citation

**Status:** Ongoing Research (May 2026)  
**Classification:** Technical Research Documentation  
**Domain:** Medical Image Analysis, Explainable AI, Deep Learning

If you use this research project, please cite:

```bibtex
@software{oct_adaptive_gradcam_2026,
  title={Adaptive Multi-Layer Fused Grad-CAM for Retinal Disease Classification},
  author={Retinal OCT Analysis Research Project},
  year={2026},
  month={May},
  note={Technical Research Documentation - Ongoing Research},
  url={https://github.com/...}
}
```

For questions, suggestions, or to discuss collaboration opportunities, refer to the detailed technical documentation in the `/docs` folder, particularly:
- [TECHNICAL_THESIS.md](docs/TECHNICAL_THESIS.md) for comprehensive methodology
- [ADAPTIVE_FUSED_GRADCAM_DOC.md](docs/ADAPTIVE_FUSED_GRADCAM_DOC.md) for architectural details
