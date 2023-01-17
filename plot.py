import numpy as np
import matplotlib.pyplot as plt

def gradient(x1, x2, y1, y2):
        m = (y2-y1) / (x2-x1)
        c = (m*x1)-y1
        sign = "+" if c>0 else "-"
        fun = f"{m}x {sign} {abs(c)}"
        return fun
def get_hyperplane_value(x, w, b, offset=0):
    return (-w[0] * x + b + offset) / w[1]

def plot_perceptron(X_train, y_train, clf):
    
    x0_1 = np.amin(X_train[:, 0])
    x0_2 = np.amax(X_train[:, 0])

    x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b)
    x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b)
    print(f"f(x) = {gradient(x0_1, x0_2, x1_1, x1_2)}")

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train, s=[100 for _ in range(len(X_train))])

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")

    x1_min = np.amin(X_train[:, 1])
    x1_max = np.amax(X_train[:, 1])
    ax.set_ylim([x1_min-0.5, x1_max+0.5])
    ax.set_title('Perceptron Plot')

def plot_svm(X_train, y_train, clf):
    x0_1 = np.amin(X_train[:, 0])
    x0_2 = np.amax(X_train[:, 0])

    x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b, 0)
    x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b, 0)
    print(f"f(x)[tengah] = {gradient(x0_1, x0_2, x1_1, x1_2)}")

    x1_1_m = get_hyperplane_value(x0_1, clf.w, clf.b, -1)
    x1_2_m = get_hyperplane_value(x0_2, clf.w, clf.b, -1)
    print(f"f(x)[bawah] = {gradient(x0_1, x0_2, x1_1_m, x1_2_m)}")

    x1_1_p = get_hyperplane_value(x0_1, clf.w, clf.b, 1)
    x1_2_p = get_hyperplane_value(x0_2, clf.w, clf.b, 1)
    print(f"f(x)[atas] = {gradient(x0_1, x0_2, x1_1_p, x1_2_p)}")

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train, s=[100 for _ in range(len(X_train))])

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
    ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
    ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

    x1_min = np.amin(X_train[:, 1])
    x1_max = np.amax(X_train[:, 1])
    ax.set_ylim([x1_min-0.5, x1_max+0.5])
    ax.set_title('SVM Plot')