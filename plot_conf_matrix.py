import numpy as np
import matplotlib.pyplot as plt
import math

conf_matrix = np.load("conf_matrix.npy")
images = np.load("images.npy")
probability = np.load("probability.npy")

target_classes = [
    'bigleaf', 'boxelder', 'cannabis', 'japanese m.', 'norway m.', 'red m.', 'silver m.', 'sugar m. '
]
fig = plt.figure()
# plt.suptitle("Visual Confusion Matrix (Highest Confidence)")
plt.ylabel("Label")
plt.xlabel("Pred")
col = 8
row = 8

for i in range(1, col*row + 1):
    r = math.floor((i-1)/row)
    c = (i-1) % col
    img = images[r][c]
    img = img.transpose(1,2,0)
    img = img*0.25 + 0.5
    fig.add_subplot(row, col, i)
    plt.title("Pr={0:.3f}".format(probability[r][c]))
    if c == 0:
        plt.ylabel("{}".format(target_classes[r]))
    if r == 7:
        plt.xlabel("{}".format(target_classes[c]))

    plt.imshow(img)


