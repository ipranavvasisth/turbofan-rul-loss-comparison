# turbofan-rul-loss-comparison

Predicting the Remaining Useful Life (RUL) of turbofan engines from sensor data,
and comparing how different loss functions change model behaviour.

## Goal
Train the same LSTM with different loss functions (MSE, MAE, MSLE, MAPE,
Tweedie, cross-entropy, and a custom asymmetric loss based on the
C-MAPSS scoring function) and compare them on identical held-out engines.

## Dataset
NASA C-MAPSS turbofan degradation dataset, subset FD001
(100 train engines, 100 test engines, one operating condition).
Download it from the NASA Prognostics Data Repository and place
`train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt` in `data/`.

## Approach
1. Build the RUL target (clipped at 125 cycles) and scale features
2. Classical baselines: ridge regression, random forest
3. LSTM in PyTorch with sliding windows
4. Swap loss functions, with all other settings and seeds fixed
5. Evaluate with RMSE, MAE, and the NASA asymmetric score

## Results
Ridge 20.8, Random Forest 17.0, LSTM (MSE, window 30, 60 epochs, seed 42) 13.6, all with test RUL clipped at 125

## Setup
    python -m venv venv
    pip install -r requirements.txt

## Course context
Built as an applied project for the NPTEL course
*Machine Learning for Core Engineering Disciplines* (IISc Bangalore).