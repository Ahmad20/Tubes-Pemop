import numpy as np
import pandas as pd

import model
import plot

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import seaborn as sns
import matplotlib.pyplot as plt

# Convert TXT to CSV
def txt2csv(data_path):
    txt_file= pd.read_csv(data_path)
    txt_file.to_csv("data.csv", index=None, )
    return pd.read_csv("data.csv", sep="\t")

# Normalizer
def norm(df):
    target_cols = 'LABEL'
    y = df[target_cols]
    y = np.where(y == 1, -1, 1)
    X = df.drop(target_cols, axis=1)
    X_normalized = StandardScaler().fit_transform(X.iloc[:,[0,1]].values)
    return np.array(X_normalized), y

# Akurasi
def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

if __name__ == '__main__':
    df = txt2csv('DATA TUGAS BESAR/DataClassification.txt')
    df['LABEL'].value_counts().plot(kind="barh", color=['red', 'blue'], title="Sebaran Data")
    X, y = norm(df)
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)
    
    # SVM Side
    svm = model.SVM()
    svm.fit(X_train, y_train)
    predictions = svm.predict(X_test)
    print(f'Akurasi Model SVM :{accuracy(y_test, predictions)}')
    sns.heatmap(confusion_matrix(y_test, predictions), annot=True, cmap='Blues').set(title="Confusion Matrix SVM")
    plt.show()
    print(classification_report(y_test, predictions))
    plot.plot_svm(X_test, y_test, svm)
    plt.show()

    print("\n"*2)

    # Perceptron Side
    perceptron = model.Perceptron()
    perceptron.fit(X_train, y_train)
    predictions = perceptron.predict(X_test)
    print(f'Akurasi Model Perceptron: {accuracy(y_test, predictions)}')
    sns.heatmap(confusion_matrix(y_test, predictions), annot=True, cmap='Blues').set(title="Confusion Matrix Perceptron")
    plt.show()
    print(classification_report(y_test, predictions))
    plot.plot_perceptron(X_test, y_test, perceptron)
    plt.show()