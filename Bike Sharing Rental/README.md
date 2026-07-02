# 🚲 Bike Sharing Rental Demand Prediction

## 📌 Project Overview

This project predicts the number of bike rentals based on weather conditions, season, date, and time using Machine Learning. The project includes complete data preprocessing, exploratory data analysis (EDA), model building, evaluation, and deployment using Streamlit.

The objective is to help bike rental companies estimate rental demand accurately so they can optimize bike availability and improve customer satisfaction.

---

## 🎯 Business Objective

The main objective of this project is to predict bike rental demand using historical data. Accurate predictions help organizations:

- Improve bike availability
- Reduce waiting time for customers
- Optimize resource allocation
- Increase operational efficiency
- Support data-driven decision making

---

## 📂 Project Structure

```
Bike_Sharing_Rental/
│
├── Dataset.csv
├── Bike Sharing Rental (4).ipynb
├── app.py
├── bike_rental_model.pkl
├── Bike_Rental_Demand_Prediction.pptx
└── README.md
```

---

## 📊 Dataset Features

The dataset contains features such as:

- Season
- Year
- Month
- Hour
- Holiday
- Weekday
- Working Day
- Weather Situation
- Temperature
- Feeling Temperature
- Humidity
- Wind Speed

### Target Variable

- **cnt** (Total Bike Rentals)

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

---

## 📈 Machine Learning Models Used

The following regression models were trained and evaluated:

- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Linear Regression

Models were evaluated using:

- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² Score

The best-performing model was saved and used for deployment.

---

## 📊 Project Workflow

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Data Preprocessing
6. Model Building
7. Model Evaluation
8. Model Selection
9. Model Deployment using Streamlit

---

## 🚀 Model Deployment

The trained model is deployed using **Streamlit**.

### Run the application

```bash
streamlit run app.py
```

Then open the local URL displayed in the terminal.

---

## 📷 Application Features

- User-friendly interface
- Input bike rental conditions
- Predict rental demand instantly
- Fast and interactive

---

## 📁 Files Included

| File | Description |
|------|-------------|
| Dataset.csv | Original dataset |
| Bike Sharing Rental (4).ipynb | Complete notebook |
| bike_rental_model.pkl | Saved trained model |
| app.py | Streamlit application |
| Bike_Rental_Demand_Prediction.pptx | Project presentation |

---

## 📌 Future Improvements

- Hyperparameter tuning
- Cross-validation
- Cloud deployment
- Real-time prediction
- Dashboard integration
- API deployment using Flask/FastAPI

---

## 👩‍💻 Author

**Sameera**

B.Tech – Computer Science and Engineering

Machine Learning Project

---

## ⭐ If you found this project useful, don't forget to star the repository!
