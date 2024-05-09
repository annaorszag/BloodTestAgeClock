# BloodTestAgeClock

Description: This workflow is for age prediction based on blood test and body metrics, related to the research of Juhász V, Ország A, Balla D, Szabó L, Sydó N, Kiss O, Csulak E, Babity M, Dohy Z, Skoda R, Becker D, Merkely B, Benczúr A, Vágó H and Kerepesi C:
[Blood Test-Based age Acceleration Is Inversely Associated with High-Volume Sports Activity](https://journals.lww.com/acsm-msse/fulltext/2024/05000/blood_test_based_age_acceleration_is_inversely.13.aspx).

Datasets: NHANES dataset split into Nhanes_train.csv training set, Nhanes_val.csv validation set and Nhanes_test.csv testing set, Hungarian athlete datasets athletes_data_1st_ex.csv and athletes_data_2nd_ex.csv, as a control of the athlete dataset, we selected a subset of the NHANES test dataset Nhanes_test_age_matched.csv whose age distribution matched the age distribution of the athlete dataset.

Requirements: Python 3.10.12, pandas 1.5.2, numpy 1.23.5, csv 1.0, lightgbm 3.3.3.99, sklearn 1.1.3, scipy 1.9.3

Run:
```bash
# Train lgbm with learning_rate 0.2, max_depth 5, num_boost_round 100, clock1
python train_lgbm.py 0.2 5 100 clock1 Nhanes_train.csv Nhanes_val.csv
# Output: lgb_model_clock1.txt
# Test clock1
python test_lgbm.py lgb_model_clock1.txt Nhanes_test_age_matched.csv athletes_data_1st_ex.csv preds_age-matched_NHANES_test_clock1.csv preds_athletes_clock1.csv
# Output: preds_age-matched_NHANES_test_clock1.csv and preds_athletes_clock1.csv
```

```bash
# Train lgbm with learning_rate 0.2, max_depth 5, num_boost_round 100, clock2
python train_lgbm.py 0.2 5 100 clock2 Nhanes_train.csv Nhanes_val.csv
# Output: lgb_model_clock2.txt
# Test clock2
python test_lgbm.py lgb_model_clock2.txt Nhanes_test_age_matched.csv athletes_data_1st_ex.csv preds_age-matched_NHANES_test_clock2.csv preds_athletes_clock2.csv
# Output: preds_age-matched_NHANES_test_clock2.csv and preds_athletes_clock2.csv
```

