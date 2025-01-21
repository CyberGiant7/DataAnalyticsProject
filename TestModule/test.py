import pickle

import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn import preprocessing
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import OrdinalEncoder


class TabNet(torch.nn.Module):
    '''
    Wrapper class for TabNetClassifier
    '''

    def __init__(self, n_d,
                 n_a,
                 n_steps,
                 gamma,
                 optimizer_fn,
                 n_independent,
                 n_shared,
                 epsilon,
                 seed,
                 lambda_sparse,
                 clip_value,
                 momentum,
                 optimizer_params,
                 scheduler_params,
                 mask_type,
                 scheduler_fn,
                 device_name,
                 output_dim,
                 batch_size,
                 num_epochs,
                 unsupervised_model,
                 cat_idxs=None,
                 cat_dims=None,
                 verbose=0):
        super(TabNet, self).__init__()

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.unsupervised_model = unsupervised_model
        self.network = TabNetClassifier(n_d=n_d,
                                        n_a=n_a,
                                        n_steps=n_steps,
                                        gamma=gamma,
                                        optimizer_fn=optimizer_fn,
                                        n_independent=n_independent,
                                        n_shared=n_shared,
                                        epsilon=epsilon,
                                        seed=seed,
                                        lambda_sparse=lambda_sparse,
                                        clip_value=clip_value,
                                        momentum=momentum,
                                        optimizer_params=optimizer_params,
                                        scheduler_params=scheduler_params,
                                        mask_type=mask_type,
                                        scheduler_fn=scheduler_fn,
                                        device_name=device_name,
                                        output_dim=output_dim,
                                        verbose=verbose,
                                        cat_idxs=cat_idxs,
                                        cat_dims=cat_dims)

    def fit_model(self, X_train, y_train, X_val, y_val, criterion):
        self.network.fit(X_train=X_train,
                         y_train=y_train,
                         eval_set=[(X_train, y_train), (X_val, y_val)],
                         eval_metric=['balanced_accuracy'],
                         patience=10,
                         batch_size=self.batch_size,
                         virtual_batch_size=128,
                         num_workers=0,
                         drop_last=True,
                         max_epochs=self.num_epochs,
                         loss_fn=criterion,
                         from_unsupervised=self.unsupervised_model)

    def predict(self, X):
        return self.network.predict(X)

    def explain(self, X):
        return self.network.explain(X)

    def feature_importances(self):
        return self.network.feature_importances_


class CustomOrdinalEncoder(OrdinalEncoder):
    def transform(self, X):
        encoded = super().transform(X)
        # Shift all values by +1 and replace unknown_value (-1) with 0
        return np.where(encoded == -1, 0, encoded + 1)

    def inverse_transform(self, X):
        # Handle the inverse transform to account for the +1 offset
        X = np.where(X == 0, -1, X - 1)
        return super().inverse_transform(X)


class TabTransformer(torch.nn.Module):
    def __init__(self, cat_dims, num_numerical, num_classes, dim_embedding=8, num_heads=2, num_layers=2, dropout=0.1):
        """
        Args:
            cat_dims: List of integers, dove ogni elemento rappresenta i valori unici di una colonna categoriale.
            num_numerical: Numero di caratteristiche numeriche.
            num_classes: Numero di classi per output.
            dim_embedding: Dimensione degli embeddings.
            num_heads: Numero di "head" nel Multi-Head Attention.
            num_layers: Numero di livelli Transformer.
            dropout: Dropout per prevenire overfitting.
        """
        super(TabTransformer, self).__init__()

        # Embeddings per features categoriali
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(cat_dim, dim_embedding) for cat_dim in cat_dims
        ])

        # Layer per le features numeriche
        self.numerical_norm = torch.nn.LayerNorm(num_numerical) if num_numerical > 0 else None

        # Transformer Encoder
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=dim_embedding,
            nhead=num_heads,
            dim_feedforward=dim_embedding * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classificatore finale
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(len(cat_dims) * dim_embedding + (num_numerical if num_numerical > 0 else 0), 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, x_cat, x_num):
        """
        Args:
            x_cat: Tensore (batch_size, num_categorical_features), indici per features categoriali.
            x_num: Tensore (batch_size, num_numerical_features), valori numerici.
        Returns:
            Logits (batch_size, num_classes).
        """
        # Embedding per features categoriali
        x_cat = x_cat.long()
        cat_embeddings = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        cat_embeddings = torch.stack(cat_embeddings, dim=1)  # (batch_size, num_categorical_features, dim_embedding)

        # Passa attraverso il Transformer
        transformed_cat = self.transformer(cat_embeddings)  # (batch_size, num_categorical_features, dim_embedding)
        transformed_cat = transformed_cat.view(transformed_cat.size(0), -1)  # Flatten per concatenare

        # Normalizzazione delle features numeriche
        if x_num is not None and self.numerical_norm is not None:
            x_num = self.numerical_norm(x_num)

        # Concatenazione
        if x_num is not None:
            x = torch.cat([transformed_cat, x_num], dim=1)
        else:
            x = transformed_cat

        # Classificatore
        logits = self.classifier(x)
        return logits

class PyTorchTabTransformer:
    def __init__(self, model, cat_idx, num_idx, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.cat_idx = cat_idx
        self.num_idx = num_idx


    def predict(self, X):
        """
        Esegue le previsioni sul modello PyTorch.
        """
        self.model.eval()  # Modalità di valutazione
        with torch.no_grad():
            # Controlla se X è un array numpy e convertilo in un tensore PyTorch
            if isinstance(X, np.ndarray):
                X = torch.tensor(X, dtype=torch.float32).to(self.device)

            # Supponi che X sia diviso in categoriale e numerico
            y_pred = self.model(X[:, self.cat_idx].long(),
                                X[:, self.num_idx])
            return torch.argmax(y_pred, dim=1).cpu().numpy()

MY_UNIQUE_ID = "TestUser"


# Output: unique ID of the team
def getName():
    return MY_UNIQUE_ID


# Input: Test dataframe
# Output: PreProcessed test dataframe
def preprocess(data, clfName):
    features = list(data.columns)
    # features_to_remove = ["label", "ts", "src_ip", "dst_ip", "dns_query", "ssl_subject", "ssl_issuer", "http_uri", "type", "http_referrer", "http_user_agent"]
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

    match clfName:
        case "knn" | "rf":
            with open("transformer/target_encoder.save", 'rb') as f:
                target_encoder: preprocessing.LabelEncoder = pickle.load(f)
            with open("transformer/transformer.save", 'rb') as f:
                transformer = pickle.load(f)
            # with open("transformer/sfs.save", "rb") as f:
            #     sfs: SequentialFeatureSelector = pickle.load(f)


            X = transformer.transform(X)
            X.columns = [col.replace("remainder__", "scale__") for col in X.columns]
            # X = sfs.transform(X)
            # X = pca.transform(X)
            y = target_encoder.transform(y)
            # return pd.concat([pd.DataFrame(X), pd.DataFrame(y, columns=["type"])], axis=1)
            return np.column_stack((X, y))

        case "svm":
            with open("transformer/target_encoder.save", 'rb') as f:
                target_encoder: preprocessing.LabelEncoder = pickle.load(f)
            with open("transformer/transformer.save", 'rb') as f:
                transformer = pickle.load(f)
            with open("transformer/nystroem_svm.save", "rb") as f:
                nystroem_svm = pickle.load(f)

            X = transformer.transform(X)
            X = nystroem_svm.transform(X)
            y = target_encoder.transform(y)
            return np.column_stack((X, y))

        case "ff" | "tb" | "tf":
            with open("transformer/target_encoder.save", 'rb') as f:
                target_encoder: preprocessing.LabelEncoder = pickle.load(f)
            with open("transformer/transformer_tb.save", 'rb') as f:
                transformer = pickle.load(f)

            X = transformer.transform(X)

            # cat_idxs = [i for i, f in enumerate(X.columns) if "cat__" in f]
            # cat_dims = [len(X[f].unique()) for i, f in enumerate(X.columns) if "cat__" in f]
            # print( list(enumerate(cat_dims)))
            y = target_encoder.transform(y)
            return np.column_stack((X, y))
        case "tf":
            with open("transformer/target_encoder.save", 'rb') as f:
                target_encoder: preprocessing.LabelEncoder = pickle.load(f)
            with open("transformer/transformer_tf.save", 'rb') as f:
                transformer = pickle.load(f)

            X = transformer.transform(X)
            y = target_encoder.transform(y)
            return np.column_stack((X, y))


# Input: Classifier name ("lr": Logistic Regression, "svc": Support Vector Classifier)
# Output: Classifier object
def load(clfName):
    match clfName:
        case "knn":
            import os
            os.environ['OMP_NUM_THREADS'] = '4'
            return pickle.load(open("models/knn2.save", 'rb'))
        case "rf":
            return pickle.load(open("models/rf.save", 'rb'))
        case "svm":
            return pickle.load(open("models/svm.save", 'rb'))
        case "ff":
            return pickle.load(open("models/ff.save", 'rb'))
        case "tb":
            return pickle.load(open("models/tabnet_model.save", 'rb'))
        case "tf":
            # return pickle.load(open("models/model_best_tf.save", 'rb'))
            # return torch.load(open("models/model_best_tf.save", 'rb'))
            return pickle.load(open("models/model_best_tf_custom.save", 'rb'))
        case _:
            return None


# Input: PreProcessed dataset, Classifier Name, Classifier Object
# Output: Performance dictionary
def predict(data, clfName, clf):
    match clfName:
        case _:
            if isinstance(data, pd.DataFrame):
                X = data.iloc[:, :-1]
                y = data.iloc[:, -1]
            elif isinstance(data, np.ndarray):
                X = data[:, :-1]
                y = data[:, -1]

            print(X.shape, y.shape)

            ypred = clf.predict(X)
            acc = accuracy_score(y, ypred)
            bacc = balanced_accuracy_score(y, ypred)
            f1 = f1_score(y, ypred, average="weighted")
            # print(confusion_matrix(y, ypred))

            perf = {"acc": acc, "bacc": bacc, "f1": f1}

            return perf


if __name__ == '__main__':
    name = getName()
    # models = ["rf", "knn", "svm"]  # , "svm", "ff", "tb", "tf"]
    models = ["svm"]  # , "svm", "ff", "tb", "tf"]
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

# tb: {'acc': 0.5901072293324109, 'bacc': 0.6103301524448705, 'f1': 0.5906097389519115}
