import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from function import makeScore, predict_displacement_LSSVM

path = "test_data"
filename = "run2.xlsx"
df = pd.read_excel(path + "/" + filename)[2:]
for col in df.columns:
    df.rename(columns={col: col.replace(" ", "")}, inplace=True)

start = 0
end = len(df)

# Predict displacement
time_step = 0.005
win_size = 1000

disp_true = predict_displacement_LSSVM(
    df=df,
    start=start,
    end=end,
    win_size=win_size,
    time_step=time_step,
    ACCDATA="A0Z",
    sigma=0.9,
    kernel="linear",
    gamma=1e50,
)

t = df["Time"][start:end]
gt = df["D0"][start:end] / 1000

score = makeScore(disp_true[start:end], gt)
print(f"score: {score}")
print(disp_true)
# print(disp_true)
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 15))
ax[0].plot(t, gt, "black", label="model")
ax[1].plot(t, disp_true[start:end], "r--", label=f"score : {score:.3f}", alpha=1)
plt.legend()
plt.show()
