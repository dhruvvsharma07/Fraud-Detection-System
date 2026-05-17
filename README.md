<div align="center">
<img src="https://img.shields.io/badge/Status-Live-brightgreen?style=for-the-badge" />
<img src="https://img.shields.io/badge/ML-RandomForest-blue?style=for-the-badge&logo=scikit-learn" />
<img src="https://img.shields.io/badge/Deployed-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit" />

 
 # 💳 Credit Card Fraud Detection System
**An end-to-end Machine Learning system that detects fraudulent transactions in real time trained, evaluated, and deployed as a live dashboard**

[![Live Demo](https://img.shields.io/badge/🔗_Live_Demo-Click_Here-0066FF?style=for-the-badge)](https://frauddetectionsystem111.streamlit.app/)
 
</div>

## 📌 What This Project Does
 
This is a **fully deployed fraud detection system**, not just a Jupyter notebook. It ingests anonymized credit card transaction data, trains a machine learning model, evaluates it rigorously, and serves real-time predictions through an interactive dashboard — all in one pipeline.
> Built to simulate what a production fraud detection pipeline might look like at a financial institution.
 
---



# 📌 What This Project Does
 
This is a **fully deployed fraud detection system**, not just a Jupyter notebook. It ingests anonymized credit card transaction data, trains a machine learning model, evaluates it rigorously, and serves real-time predictions through an interactive dashboard — all in one pipeline.
 
> Built to simulate what a production fraud detection pipeline might look like at a financial institution.

## 🖥️ Live Demo
 
🔗 **[frauddetectionsystem111.streamlit.app](https://frauddetectionsystem111.streamlit.app/)**
 
Try selecting any transaction from the dataset and see the model classify it as **Fraud** or **Legit** — with a confidence score — in real time.
 
---
 
## ✨ Features
 
| Feature | Description |
|--------|-------------|
| 🔮 **Real-Time Prediction** | Select a transaction → instant Fraud / Legit classification with confidence score |
| 📊 **Interactive Dashboard** | Class distribution, feature histograms, correlation heatmap |
| 🤖 **ML Pipeline** | StandardScaler + RandomForestClassifier with class imbalance handling |
| 📈 **Model Evaluation** | Confusion matrix, ROC curve, and feature importance charts |
| ⚡ **Deployed & Accessible** | Fully live — no local setup required |
 
---
 
## 🧪 Tech Stack
 
```
Frontend / UI     →  Streamlit
ML / Backend      →  Scikit-learn (RandomForestClassifier, StandardScaler)
Data Handling     →  Pandas, NumPy
Visualization     →  Matplotlib
Dataset Source    →  OpenML — ULB Credit Card Fraud Dataset
```
 
---
 
## 🗂️ Project Structure
 
```bash
Fraud-Detection-System/
│
├── app.py               # Main Streamlit application (prediction + dashboard)
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```
 
---
 
## 🧩 Dataset
 
- **Source**: ULB Machine Learning Group, accessed via [OpenML](https://www.openml.org/)
- **Size**: 284,807 transactions | **Fraud rate**: ~0.17% (highly imbalanced)
- **Features**: 30 total — `Time`, `Amount`, and `V1–V28` (PCA-anonymized for privacy)
- **Target**:
  - `0` → Legitimate transaction
  - `1` → Fraudulent transaction
---
 
## ⚙️ ML Pipeline
 
```
Raw Data
   ↓
StandardScaler (normalize Amount & Time)
   ↓
RandomForestClassifier
   ├── class_weight='balanced'   ← handles severe class imbalance
   └── trained on full dataset
   ↓
Evaluation: Confusion Matrix · ROC-AUC · Feature Importance
   ↓
Streamlit App → Real-Time Inference
```
 
---
 
## 🏆 Key Highlights
 
- ✅ **End-to-end system** — not just a model, but a full data-to-deployment pipeline
- ✅ **Handles real-world constraints** — class imbalance, large dataset, deployment inconsistencies
- ✅ **Live and accessible** — anyone can use it without installing anything
- ✅ **Production mindset** — structured code, reproducible pipeline, clean UI
---
 
## 🧠 What I Learned
 
- Debugging ML deployment issues in cloud environments (Streamlit Cloud)
- Managing class imbalance without oversampling (weighted learning)
- Handling dataset inconsistencies across local and remote environments
- Building and deploying a production-ready ML pipeline end-to-end
---
 
## 🚀 Planned Improvements
 
- [ ] Manual transaction input form (enter custom values for prediction)
- [ ] Enhanced UI/UX with better dashboard theming
- [ ] Experiment with XGBoost / LightGBM for performance gains
- [ ] REST API endpoint for programmatic predictions
---
 
## 🙌 Acknowledgements
 
- Dataset by the **ULB Machine Learning Group** — accessed via [OpenML](https://www.openml.org/d/1597)
- Deployment powered by **Streamlit Community Cloud**
---
 
## 📬 Connect
 
Found this interesting? Feel free to ⭐ the repo or reach out!
 
<div align="center">
---
*Built with curiosity, debugged with patience.*
 
</div>
 


