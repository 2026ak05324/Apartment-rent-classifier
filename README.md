# Apartment Rent Classification (5 ML Models + Streamlit App)

## a. Problem Statement
Classify whether a rental listing's **monthly rent is High (above the dataset median) or Low**, using listing attributes
(bedrooms, bathrooms, size, location, amenities, pet policy, etc.). Five classifiers are trained on the same data and
compared with Accuracy, AUC, Precision, Recall, F1 and MCC. A Streamlit app lets users upload a test CSV, pick a model and
view metrics, a confusion matrix and a classification report.

## b. Dataset Description
- **Name:** Apartments for Rent Classified (UCI Machine Learning Repository), file `apartments_for_rent_classified.csv` (5,000-row sample).
- **Size:** 4,980 monthly-rent listings (5,000 raw rows, 22 columns), 20 engineered numeric features.
- **Target:** `price_class` = 1 if `price` > median monthly price ($1,058.50), else 0 (balanced by construction).
- **Cleaning:** the literal string `null` is treated as missing; kept rows with `price_type == Monthly` and price > 0; dropped text/ID columns (id, title, body, address, source, time, price_display, currency).
- **Features:** bathrooms, bedrooms, square_feet, latitude, longitude, has_photo, pets_allowed, n_amenities,
  state/city frequency encodings (learned from the training split only), and 10 amenity flags.
- **Split:** 80/20 stratified train/test (random_state=42) (3,984 train / 996 test). The test split is saved as `test_data.csv`.
- **Preprocessing (inside each saved model pipeline):** median imputation + standard scaling.

## c. GitHub Repository Link
<FILL: https://github.com/<user>/<repo>>

Live app: <FILL: https://<app-name>.streamlit.app>

## d. Models Used

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.6376 | 0.6920 | 0.6553 | 0.5803 | 0.6155 | 0.2769 |
| Decision Tree | 0.8042 | 0.8900 | 0.8548 | 0.7329 | 0.7892 | 0.6147 |
| KNN | 0.7279 | 0.8008 | 0.7574 | 0.6707 | 0.7114 | 0.4588 |
| Naive Bayes | 0.6265 | 0.7287 | 0.7386 | 0.3916 | 0.5118 | 0.2866 |
| Support Vector Machine | 0.7761 | 0.8592 | 0.8161 | 0.7129 | 0.7610 | 0.5567 |

Metrics are computed on the 996-row held-out test set (`test_data.csv`); printed by `train.py`.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Weakest overall (MCC 0.28). Rent depends non-linearly on location (lat/long, state) and listings are mostly similar 1-bedroom units, so a linear boundary underfits. |
| Decision Tree | Best on every metric (Acc 0.80, AUC 0.89, MCC 0.61). Splits on latitude/longitude and state effectively carve out expensive vs cheap regions; max_depth=8 and min_samples_leaf=10 keep overfitting in check. |
| KNN | Solid but behind tree and SVM (MCC 0.46). Distance on scaled lat/long finds similar markets, but the many binary amenity features add noise to the distance. |
| Naive Bayes | Poor recall (0.39) with decent precision (0.74): it is conservative about predicting High. The independence assumption fails because location and amenity features are correlated, and Gaussian likelihoods suit binary flags badly. |
| Support Vector Machine | Second best (AUC 0.86, MCC 0.56). The RBF kernel captures non-linear geographic effects, but it trails the tree and is slower to train and score. |

**Overall winner:** **Decision Tree** - highest Accuracy, AUC, Precision, F1 and MCC on the held-out test set, with the SVM a clear second.

## Project Structure
```
app.py            Streamlit app (loads saved models, never retrains)
train.py          preprocessing + trains the 5 models, writes model/*.joblib and test_data.csv
requirements.txt
test_data.csv     held-out test split
model/            logistic_regression / decision_tree / knn / naive_bayes / support_vector_machine .joblib
```

## Run Locally
```
pip install -r requirements.txt
python train.py apartments_for_rent_classified.csv
streamlit run app.py
```
