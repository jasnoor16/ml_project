#  **How to Use This Project**

##  **Preprocessing**
Before training, you **must preprocess the data**:
```bash
python src/preprocess.py
```
This will **clean the datasets**, apply feature transformations, and save `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`.

---

##  **Training the Model**
To train the model, run:
```bash
python src/train.py
```
This will train:
- **Linear Regression**
- **Random Forest**
- **Decision Tree**

All trained models will be **saved in the `models/` folder**.

---

##  **Evaluating the Model**
To assess model performance:
```bash
python src/evaluate.py
```
The script will output **MAE, RMSE, R² Score** for all models.

---

##  **Making Predictions**
To generate predictions:
```bash
python src/predict.py
```
This will:
- Load `X_test.npy`
- Use the **best-performing model** (default: `linear regression`)
- Compare **Predicted vs. Actual Values**

---

## **Files & Outputs**
| File                   | Purpose |
|------------------------|---------|
| `data/raw/`           | Stores original datasets |
| `data/processed/`     | Contains cleaned datasets |
| `models/`             | Stores trained ML models |
| `configs/`            | Configuration files (YAML) |
| `predictions/`        | Saves final predictions |

---

##  **Changing Model Configurations**
Modify `configs/train_config.yaml` to change hyperparameters:
```yaml
random_forest:
  n_estimators: 100
  max_depth: null
  random_state: 42
```
To change the **model used for predictions**, edit `configs/predict_config.yaml`:
```yaml
default_model: "linear"
output_directory: "./p
