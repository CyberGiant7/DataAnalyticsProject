import pickle

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

from utils import *

MY_UNIQUE_ID = "SigmoidSquad"


# Output: unique ID of the team
def getName():
    return MY_UNIQUE_ID


# Input: Test dataframe
# Output: PreProcessed test dataframe
def preprocess(data, clfName):
    features = list(data.columns)
    features_to_remove = ["label", "type", "ts", "http_referrer"]
    features = [feature for feature in features if feature not in features_to_remove]
    data = data[features + ["type"]]

    # Converte i valori in numeri, sostituendo quelli non validi con NaN
    data.loc[:, "src_bytes"] = pd.to_numeric(data["src_bytes"], errors='coerce')
    # Filtra le righe con NaN (valori non convertibili)
    data = data.dropna(subset=["src_bytes"])
    # Converte i valori rimasti in interi
    data.loc[:, "src_bytes"] = data["src_bytes"].astype(int)

    X = data[features]
    y = data["type"]

    with open("transformer/target_encoder.save", 'rb') as f:
        target_encoder: preprocessing.LabelEncoder = pickle.load(f)
    y = target_encoder.transform(y)

    match clfName:
        case "knn":
            with open("transformer/transformer_knn.save", 'rb') as f:
                transformer = pickle.load(f)
            X = transformer.transform(X)
            # concat X and y with pandas dataframe
            return pd.concat([X, pd.DataFrame(y)], axis=1)
        case "rf":
            with open("transformer/transformer_rf.save", 'rb') as f:
                transformer = pickle.load(f)
            X = transformer.transform(X)
            return pd.concat([X, pd.DataFrame(y)], axis=1)
        case "svm":
            with open("transformer/transformer_svm.save", 'rb') as f:
                transformer = pickle.load(f)
            # with open("transformer/nystroem_svm.save", "rb") as f:
            #     nystroem_svm = pickle.load(f)
            X = transformer.transform(X)
            # X = nystroem_svm.transform(X)
            y_sparse = csr_matrix(y.reshape(-1, 1))
            return hstack([X, y_sparse])
        case "ff":
            with open("transformer/transformer_ff.save", 'rb') as f:
                transformer = pickle.load(f)
            X = transformer.transform(X)
            return pd.concat([X, pd.DataFrame(y)], axis=1)
        case "tb":
            with open("transformer/transformer_tb.save", 'rb') as f:
                transformer = pickle.load(f)
            X = transformer.transform(X)
            return np.column_stack((X, y))
        case "tf":
            with open("transformer/transformer_tf.save", 'rb') as f:
                transformer = pickle.load(f)
            X = transformer.transform(X)
            return np.column_stack((X, y))


# Output: Classifier object
def load(clfName):

    match clfName:
        case "knn":
            model: KNeighborsClassifier = pickle.load(open("models/knn.save", 'rb'))
        case "rf":
            model: RandomForestClassifier = pickle.load(open("models/rf.save", 'rb'))
        case "svm":
            model: SGDClassifier = pickle.load(open("models/svm.save", 'rb'))
        case "ff":
            model: PyTorchFeedForwardWrapper = pickle.load(open("models/ff.save", 'rb'))
        case "tb":
            model: TabNet = pickle.load(open("models/tb.save", 'rb'))
        case "tf":
            model: PyTorchTabTransformerWrapper = pickle.load(open("models/tf.save", 'rb'))
        case _:
            return None
    return model


# Input: PreProcessed dataset, Classifier Name, Classifier Object
# Output: Performance dictionary
def predict(data, clfName, clf):
    if isinstance(data, pd.DataFrame):
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
    elif isinstance(data, np.ndarray):
        X = data[:, :-1]
        y = data[:, -1]
    elif isinstance(data, csr_matrix):
        X = data[:, :-1]
        y = data[:, -1].toarray()
    else:
        return None

    match clfName:
        case "ff":
            X = X.to_numpy()
        case _:
            pass
    ypred = clf.predict(X)
    acc = accuracy_score(y, ypred)
    bacc = balanced_accuracy_score(y, ypred)
    f1 = f1_score(y, ypred, average="weighted")
    return {"acc": acc, "bacc": bacc, "f1": f1}


# if __name__ == '__main__':
#     name = getName()
#     models = ["knn","rf", "svm", "ff", "tb", "tf"]
#     data = pd.read_csv("../TrainingModule/dataset/val_dataset.csv", sep=",", low_memory=False)
#
#     for model in models:
#         dfProcessed = preprocess(data, model)
#         clf = load(model)
#         perf = predict(dfProcessed, model, clf)
#         print(f"{model}: {perf}")

