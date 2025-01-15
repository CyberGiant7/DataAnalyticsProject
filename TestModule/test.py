import pandas as pd
import pickle
from sklearn import preprocessing
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

MY_UNIQUE_ID = "TestUser"

# Output: unique ID of the team
def getName():
    return MY_UNIQUE_ID


# Input: Test dataframe
# Output: PreProcessed test dataframe
def preprocess(data, clfName):
    features = list(data.columns)
    features_to_remove = ["label", "ts", "src_ip", "dst_ip", "dns_query", "ssl_subject", "ssl_issuer", "http_uri", "type", "http_referrer"]
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

    match clfName:
        case "knn" | "rf":
            with open("transformer/target_encoder.save", 'rb') as f:
                target_encoder: preprocessing.LabelEncoder = pickle.load(f)
            with open("transformer/transformer.save", 'rb') as f:
                transformer = pickle.load(f)
            with open("transformer/sfs.save", "rb") as f:
                sfs: SequentialFeatureSelector = pickle.load(f)
            # with open("transformer/pca.save", 'rb') as f:
            #     pca = pickle.load(f)

            X = transformer.transform(X)
            X.columns = [col.replace("remainder__", "scale__") for col in X.columns]
            X = sfs.transform(X)
            # X = pca.transform(X)
            y = target_encoder.transform(y)
            return pd.concat([pd.DataFrame(X), pd.DataFrame(y, columns=["type"])], axis=1)
        case "svm" | "ff" | "tb" | "tf":
            pass


# Input: Classifier name ("lr": Logistic Regression, "svc": Support Vector Classifier)
# Output: Classifier object
def load(clfName):
    match clfName:
        case "knn":
            import os
            os.environ['OMP_NUM_THREADS'] = '4'
            return pickle.load(open("models/knn.save", 'rb'))
        case "rf":
            return pickle.load(open("models/rf.save", 'rb'))
        case "svm":
            return pickle.load(open("models/svm.save", 'rb'))
        case "ff":
            return pickle.load(open("models/ff.save", 'rb'))
        case "tb":
            return pickle.load(open("models/tb.save", 'rb'))
        case "tf":
            return pickle.load(open("models/tf.save", 'rb'))
        case _:
            return None


# Input: PreProcessed dataset, Classifier Name, Classifier Object
# Output: Performance dictionary
def predict(data, clfName, clf):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    ypred = clf.predict(X)
    acc = accuracy_score(y, ypred)
    bacc = balanced_accuracy_score(y, ypred)
    f1 = f1_score(y, ypred, average="weighted")
    # print(confusion_matrix(y, ypred))

    perf = {"acc": acc, "bacc": bacc, "f1": f1}

    return perf


if __name__ == '__main__':
    name = getName()
    models = ["rf", "svm", "knn"]  # , "svm", "ff", "tb", "tf"]
    # models = ["rf"]  # , "svm", "ff", "tb", "tf"]
    # data = pd.read_csv("../TrainingModule/dataset/test_dataset.csv", sep=",", low_memory=False)
    # import os
    # os.chdir("./TestModule")
    data = pd.read_csv("../TrainingModule/dataset/test_dataset.csv", sep=",", low_memory=False)

    for model in models:
        dfProcessed = preprocess(data, model)
        # print(dfProcessed.head(10))
        clf = load(model)
        perf = predict(dfProcessed, model, clf)
        print(f"{model}: {perf}")


# knn: {'acc': 0.7951223210435788, 'bacc': 0.7194376510067115, 'f1': 0.7888996709679958} knn con minmax
# rf: {'acc': 0.9689305023146941, 'bacc': 0.9571513403643335, 'f1': 0.9691014407258134} rf con minmax

# knn: {'acc': 0.7944873793492322, 'bacc': 0.7398609558964525, 'f1': 0.787795231400687} senza minmax
# rf: {'acc': 0.9700534962069342, 'bacc': 0.9578639539789071, 'f1': 0.9702047848932288} senza minmax

