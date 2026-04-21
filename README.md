# 💳 Credit Card Fraud Detection System

An end-to-end Machine Learning project that detects fraudulent transactions using real-world financial data and is deployed as a live interactive web application.



## 🚀 Live Demo

🔗 https://frauddetectionsystem111.streamlit.app/



## 🧠 Project Overview

This project focuses on detecting fraudulent credit card transactions using supervised machine learning techniques. It goes beyond model training by integrating:

* Data preprocessing
* Feature analysis
* Model building
* Real-time prediction
* Interactive dashboard deployment



## ⚙️ Features

* 🔮 **Real-time Fraud Prediction**

  * Select any transaction and instantly classify it as Fraud / Legit
  * Displays confidence score

* 📊 **Interactive Analysis Dashboard**

  * Class distribution visualization
  * Feature distributions (histograms)
  * Correlation analysis

* 🤖 **Machine Learning Pipeline**

  * StandardScaler + RandomForestClassifier
  * Handles class imbalance using weighted learning

* 📈 **Model Evaluation**

  * Confusion Matrix
  * ROC Curve
  * Feature Importance visualization



## 🧪 Tech Stack

* **Frontend / App**: Streamlit
* **Backend / ML**: Scikit-learn
* **Data Handling**: Pandas, NumPy
* **Visualization**: Matplotlib
* **Dataset Source**: OpenML (Credit Card Fraud Dataset)



## 🗂️ Project Structure

```bash
Fraud-Detection-System/
│
├── app.py                # Main Streamlit application
├── requirements.txt      # Dependencies
├── README.md             # Project documentation
```



## 🧩 Dataset

* The dataset contains anonymized credit card transactions.
* Features are PCA-transformed (V1–V28) for privacy.
* Target variable:

  * `0` → Legit transaction
  * `1` → Fraudulent transaction



## ⚡ Key Highlights

* End-to-end ML system (not just a notebook)
* Fully deployed and accessible online
* Handles real-world constraints like:

  * Large dataset limitations
  * Deployment issues
  * Data pipeline inconsistencies



## 🧠 Learnings

Through this project, I gained hands-on experience in:

* Debugging real-world ML deployment issues
* Handling dataset inconsistencies across platforms
* Building production-ready ML pipelines
* Deploying interactive ML applications



## 🚀 Future Improvements

* Add manual transaction input form
* Improve UI/UX with dashboard styling
* Optimize model performance further
* Add API-based prediction system



## 🙌 Acknowledgements

* Dataset originally sourced from ULB Machine Learning Group
* Accessed via OpenML



## 📬 Connect

If you found this project interesting, feel free to connect or reach out!


