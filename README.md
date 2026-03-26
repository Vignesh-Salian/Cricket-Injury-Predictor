# 🏏 Cricket Player Injury Risk Prediction System

A **Machine Learning-based web application** that predicts injury risk levels for cricket players using the **Random Forest algorithm**.

The system analyzes:

* Training workload
* Fatigue level
* Sleep hours
* Fitness score
* Injury history

and provides **real-time injury risk prediction** with:

* Confidence score
* Probability distribution chart
* Personalized injury prevention recommendations

---

# 🚀 Key Features

* Real-time injury prediction using **Random Forest ML model**
* Interactive **Streamlit-based dashboard**
* Probability distribution chart for **Low / Medium / High risk levels**
* Personalized **injury prevention recommendations**
* Feature engineering using **workload ratio & recovery efficiency**
* Prediction **confidence score with probability breakdown**

---

# 📂 Project Structure

```bash
cricket-injury-prediction/

├── app.py                     # Streamlit web application
├── prediction.ipynb           # Model training & experimentation
├── cricket_player_data_fixed.csv   # Training dataset
├── injury_risk_model.pkl      # Trained ML model (Random Forest)
├── model_feature_info.pkl     # Feature names & class labels
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

---

# ⚙️ Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

After running, open the **local Streamlit server** in your browser.

---

# 🛠️ Technologies Used

* Python
* Streamlit
* Scikit-learn
* Pandas
* NumPy
* Joblib

---

# 📊 Machine Learning Model

The system uses a **Random Forest Classifier** trained on cricket player workload and fitness data to classify injury risk into:

* **Low Risk**
* **Medium Risk**
* **High Risk**

The model also outputs **probability scores** for each risk category.

---

# 📌 Future Improvements

* Add real cricket player dataset
* Integrate wearable sensor data
* Deploy the application online
* Improve dashboard visualization

---

# 👨‍💻 Author

## 👨‍💻 Author

<p align="left">
  <a href="https://github.com/Vignesh-Salian">
    <img src="https://img.shields.io/badge/VIGNESH%20SALIAN-DEVELOPER-0078D4?style=for-the-badge&logo=github&logoColor=white" alt="Vignesh Salian" />
  </a>
  <a href="https://github.com/Vignesh-Salian">
    <img src="https://img.shields.io/github/followers/Vignesh-Salian?label=Follow&style=for-the-badge&logo=github" alt="Follow Vignesh Salian" />
  </a>
</p>
