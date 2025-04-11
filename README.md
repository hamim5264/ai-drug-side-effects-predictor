# 💊 AI-Driven Drug Side Effect Predictor

An AI-powered web application that predicts possible **drug side effects** based on the **drug name** and **medical condition** using machine learning.  
Supports both **international** and **Bangladeshi drugs** (e.g., *Napa, Monas, Losectil*), making it more meaningful for local users. 🌍🇧🇩

---

## 🔬 Project Objective

To help users understand potential side effects of drugs by:
- Training a machine learning model on real-world data
- Combining international + Bangladeshi drug datasets
- Creating a user-friendly web application using **Flask**

---

## 🚀 Features

✅ Predicts side effects for both **generic** and **brand-name** drugs  
✅ Accepts input via a clean web form  
✅ Displays predictions clearly based on your input  
✅ Bangladeshi drug brands like *Napa, Monas, Secrin* supported  
✅ Lightweight and fully local ML prediction

---

## 🛠 Tech Stack

- **Python**
- **Pandas**, **scikit-learn**, **XGBoost**
- **Flask** (for the web app)
- **HTML/CSS** (frontend)
- Trained and tested using **real drug data**

---

## 📁 Project Structure

├── data/ # Datasets (BD + global), preprocessed files ├── models/ # Trained ML model + encoders ├── web_app/ # Flask application │ └── templates/ # HTML UI ├── data_preprocessing.py # Preprocessing & encoding ├── model_training.py # Model training & evaluation └── README.md # Project description


---

## 💡 How to Run Locally

1. Clone this repo  
2. Create a virtual environment  
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
Train or use existing model
📸 Screenshot
![Screenshot 2025-04-12 013319](https://github.com/user-attachments/assets/4c23223a-6f45-4168-b3da-94dbf4dac76e)
![Screenshot 2025-04-12 013309](https://github.com/user-attachments/assets/5b508b8e-8946-464b-b77b-24821b76b99a)

📢 License
Feel free to use, remix, and improve with credit.
