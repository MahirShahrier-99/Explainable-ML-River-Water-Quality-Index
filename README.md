# Turag River Water Quality Index: Explainable Machine Learning Analysis

Machine learning pipeline for water quality assessment of the Turag River, Dhaka, Bangladesh. The analysis models the Water Quality Index (WQI), explains its drivers, identifies pollution sources, and forecasts future conditions. Monitoring records come from the Bangladesh Department of Environment (DoE), covering 2011 to 2023.

## What the pipeline does

1. **Feature engineering.** Builds ratio features (BOD/COD, DO deficit, TDS/EC, Cl/TDS), spatial features, temporal encodings, and interaction terms.
2. **Time series decomposition.** STL decomposition and seasonal analysis of WQI dynamics.
3. **Model benchmarking.** Random Forest, XGBoost, Gradient Boosting, and a regularized deep neural network. Models are compared on RMSE and R².
4. **Multi-scale SHAP explainability.** Global, seasonal, and location-level attribution of WQI drivers.
5. **Pollution source identification.** Parameter signature analysis classifies samples into textile dyeing, domestic sewage, mixed urban, and diffuse non-point sources.
6. **Parameter-wise WQI decomposition.** Quantifies each parameter's contribution to the composite index.
7. **Forecasting.** Monthly WQI projections from 2024 to 2035 with 95% confidence intervals.
8. **Scenario analysis.** Intervention scenarios including ETP enforcement (30% COD/BOD reduction at industrial sites) and sewage treatment (40% BOD reduction), with cost-benefit assessment.
9. **Temporal validation.** Time-based train-test splits confirm model robustness.

## Repository structure

```
.
├── Turag_WQI_Analysis.ipynb        # Main analysis notebook (run cells in order)
├── src/
│   └── turag_wqi_pipeline.py       # Standalone script version of the notebook
├── figures/                        # Final publication-quality figures (300 dpi)
│   ├── Figure1_Model_Performance_Complete.png
│   ├── Figure2_SHAP_Analysis_Complete.png
│   ├── Figure3_Source_Identification.png
│   ├── Figure4_Seasonal_Patterns.png
│   ├── Figure5_Future_Forecasts.png
│   ├── Figure6_Scenario_Analysis.png
│   └── Figure7_Summary_Dashboard.png
├── data/                           # Place the input dataset here (not tracked)
├── requirements.txt
├── LICENSE
└── .gitignore
```

## Figures

| Figure | Content |
|---|---|
| 1 | Model performance, validation, and feature importance |
| 2 | Multi-scale SHAP analysis and parameter-wise WQI decomposition |
| 3 | Pollution source identification and validation |
| 4 | Seasonal patterns and temporal trends (2011 to 2023) |
| 5 | Monthly WQI forecasts, 2024 to 2035, with 95% confidence intervals |
| 6 | Intervention scenario analysis and cost-benefit assessment |
| 7 | Comprehensive results summary and decision support dashboard |

## Getting started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Python 3.10 or later is recommended.

### 2. Add the data

Place `Turag_River_WQI_Bangladesh_DoE_Results.xlsx` in the `data/` folder. The pipeline reads the `WQI_Summary` sheet. The dataset is not distributed with this repository. Contact the author for access.

### 3. Run the analysis

Notebook:

```bash
jupyter notebook Turag_WQI_Analysis.ipynb
```

Or the standalone script from the repository root:

```bash
python src/turag_wqi_pipeline.py
```

All figures are written to `figures/` at 300 dpi.

## Notes on reproducibility

Tree-based models use `random_state=42`. Forecast confidence intervals use resampling, so bands can vary slightly between runs. The figures in `figures/` are the final versions from the reference run.

## Author

Abdulla Al Kafy

## License

MIT. See [LICENSE](LICENSE).
