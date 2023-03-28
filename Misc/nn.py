import numpy as np
import pandas as pd
import torch
from torch import nn
from matplotlib import pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from skorch import NeuralNetBinaryClassifier

np.random.seed(42)
torch.manual_seed(42)
pd.set_option("display.max_rows", 80)
pd.set_option("display.width", 1920)
pd.set_option("display.float_format", "{:20,.2f}".format)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
plt.rcParams["figure.dpi"] = 150


class LinearBNReLuBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(LinearBNReLuBlock, self).__init__()
        self.linear = nn.Linear(in_features=in_feature, out_features=out_feature)
        self.bn = nn.BatchNorm1d(out_feature)
        self.relu = nn.LeakyReLU()

    def forward(self, input):
        y = self.linear(input)
        y = self.bn(y)
        y = self.relu(y)
        return y


def main():
    df = pd.read_csv("./normalized_nybnb.csv")
    X, y = (
        df.drop(columns=["High Review Score"]).values.astype(np.float32),
        df["High Review Score"].values.astype(np.float32),
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.1,
        random_state=42,
        stratify=y,
    )

    target_names = ["Low Review Score", "High Review Score"]
    n_features = X_train.shape[1]

    net = torch.nn.Sequential(
        LinearBNReLuBlock(n_features, n_features * 2),
        LinearBNReLuBlock(n_features * 2, n_features * 4),
        # LinearBNReLuBlock(n_features * 4, n_features * 4),
        # LinearBNReLuBlock(n_features * 4, n_features * 4),
        # LinearBNReLuBlock(n_features * 4, n_features * 2),
        # LinearBNReLuBlock(n_features * 2, n_features),
        # LinearBNReLuBlock(n_features, n_features // 2),
        # LinearBNReLuBlock(n_features // 2, n_features // 4),
        LinearBNReLuBlock(n_features * 4, n_features),
        torch.nn.Linear(n_features, 1),
    )

    cls = NeuralNetBinaryClassifier(
        net,
        criterion=torch.nn.BCEWithLogitsLoss,
        optimizer=torch.optim.Adam,
        max_epochs=5,
        batch_size=256,
        device="cuda",
    )

    cls.fit(X_train, y_train)

    pred = cls.predict(X_test)
    proba = cls.predict_proba(X_test)
    print(proba[:10])
    print("Accuracy : ", accuracy_score(y_test, pred))
    print("Balanced Accuracy : ", balanced_accuracy_score(y_test, pred))
    print(cm := confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, target_names=target_names))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot()
    # disp.ax_.set_title('')


if __name__ == "__main__":
    main()
