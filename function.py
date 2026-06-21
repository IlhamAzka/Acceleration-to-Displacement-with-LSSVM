import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from tqdm import tqdm

from lssvm import LSSVMRegression


def removeMean(series):
    return series - np.mean(series)


def initSOCDConstant(size):

    socd_constant = [1, -2, 1]
    matrix = np.zeros((size, size + 2), dtype=int)
    for i in range(size):
        matrix[i][(i + 1) - 1] = socd_constant[0]
        matrix[i][(i + 1)] = socd_constant[1]
        matrix[i][(i + 1) + 1] = socd_constant[2]

    return matrix


def onlyMiddleElement(matrix, data_size, mid_pos):

    middle_only = np.zeros((data_size,), dtype=float)
    for i in range(data_size):
        middle_only[i] = matrix[i][mid_pos + 1]

    return middle_only


def predict_displacement_LSSVM(
    df: pd.DataFrame,
    start: int,
    end: int,
    win_size: int,
    time_step: float,
    ACCDATA: str,
    sigma: float,
    kernel: str,
    gamma: float,
):

    data_size = end - start

    # 1. select the window length m, and choose gamma value (for regularization)

    h = win_size
    m = 2 * h + 1
    w_len = m

    # 2. remove the mean of the acceleration vector

    acc_sensor = removeMean(df[ACCDATA][start:end]) * 9.81 * (time_step**2)
    y = np.pad(acc_sensor, (win_size, win_size), "constant", constant_values=(0, 0))

    # 3. place the center of time window at t = t_i

    disp_all_wlen = np.zeros((data_size, m + 2), dtype=float)
    X = initSOCDConstant(m)
    Xt = X.T
    y = np.reshape(y, (-1, 1))
    y = y.ravel()

    # start+h -> back to initial array start index before padding
    # end+h   -> back to initial array end index before padding

    regr = make_pipeline(
        LSSVMRegression(sigma=sigma, kernel=kernel, gamma=gamma, m=w_len, x_init=X)
    )

    # calculateDisp(regr, win_size, data_size, disp_all_wlen, X, y)
    for i in tqdm(np.arange(h, data_size + h, 1)):
        regr.fit(X, y[i - h : i + h + 1])
        disp_all_wlen[i - h] = np.dot(Xt, regr[0].coef_)

    disp_true = onlyMiddleElement(disp_all_wlen, data_size, h)
    return disp_true


def makeScore(y_predict, y_test):
    u = np.sum((y_test - y_predict) ** 2)
    v = np.sum((y_test - y_test.mean()) ** 2)

    score = 1 - u / v
    return score
