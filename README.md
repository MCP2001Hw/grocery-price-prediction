# Grocery Price Prediction with Machine Learning

Group coursework for F20DL Data Mining & Machine Learning — Heriot-Watt University (2024–25).  
5-person team. **My contribution: linear regression implemented from scratch (`Linear_Regression_Model.py`) and the data preprocessing pipeline.**

Predicts Tesco grocery prices per 100g using nutritional information (calories, fat, carbohydrates, sugars, fibre, protein, salt). Four models were implemented and compared across the team.

---

## Models

| Model | File | Author |
|---|---|---|
| Linear Regression (from scratch) | `Linear_Regression_Model.py` | Me |
| Decision Tree | `Decision_Tree_Model.py` | Team |
| K-Nearest Neighbours | `KNearest.py` | Team |
| CNN (image-based) | `CNN/` | Team |

---

## My Contribution

**Linear regression built without sklearn** — implemented gradient descent manually using numpy:
- Data loading, shuffling, and 67/33 train/test split
- Feature normalisation using training set mean and standard deviation (no data leakage)
- Weight and bias initialisation
- Gradient descent optimisation loop
- RMSE and MAE evaluation on test set

Features used: calories, fat, carbohydrates, sugars, fibre, protein, salt (per 100g)

---

## Running

```bash
pip install numpy pandas
python Linear_Regression_Model.py
```

Requires `Data.csv` in the same directory (Tesco groceries dataset — not included in this repo due to licensing).

For the CNN component:
```bash
cd CNN
python install.py
python "Image CNN 2 BatchNorm.py"
```

---

## Dependencies

- `numpy`
- `pandas`
- `tensorflow` / `keras` (CNN only)
- `scikit-learn` (Decision Tree and KNN only)
