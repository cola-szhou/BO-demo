import numpy as np

np.random.seed(0)
import seaborn as sns

sns.set()
import matplotlib.pyplot as plt

s1_data = np.asarray((0.0162, 0.0199, 0.0599, 0.1307, 0.1438, 0.5469, 0.0410, 0.0416)).reshape(1, 8)
s2_data = np.asarray((0.1211, 0.7836, 0.0113, 0.0167, 0.0093, 0.0112, 0.0468, 0)).reshape(1, -1)
sentence_weight = np.asarray((0.0391, 0.0617, 0.0481, 0.0386, 0.0817, 0.0965, 0.0934, 0.0960, 0.0975,
                              0.0877, 0.0871, 0.0961, 0.0764)).reshape(1, -1)
attention = np.asarray((0.026, 0.028, 0.11, 0.008, 0.008, 0.012, 0.078, 0.001)).reshape(1,-1)
f, ax = plt.subplots(figsize=(9, 6))
ax = sns.heatmap(data=attention, cmap='Blues', annot=attention, vmin=0, vmax=0.2)
plt.show()
