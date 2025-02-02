import numpy as np
import torch
from sklearn.preprocessing import OrdinalEncoder
from torch import nn


class CustomOrdinalEncoder(OrdinalEncoder):
    def transform(self, X):
        encoded = super().transform(X)
        # Shift all values by +1 and replace unknown_value (-1) with 0
        return np.where(encoded == -1, 0, encoded + 1)

    def inverse_transform(self, X):
        # Handle the inverse transform to account for the +1 offset
        X = np.where(X == 0, -1, X - 1)
        return super().inverse_transform(X)


class PyTorchFeedForwardWrapper:
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


class PyTorchTabTransformerWrapper:
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


class FeedForwardPlus(nn.Module):
    def __init__(self, cat_dims, num_numerical, num_categorical, num_classes, hidden_size, depth=1, batch_norm=False, drop=0, dim_embedding=8, ):
        super(FeedForwardPlus, self).__init__()

        # Initialize embeddings for categorical features
        self.embeddings = nn.ModuleList([
            nn.Embedding(cat_dim, dim_embedding) for cat_dim in cat_dims
        ])

        # Layer normalization for numerical features
        self.numerical_norm = nn.LayerNorm(num_numerical)

        model = []
        # Input layer
        model += [nn.Linear(num_categorical * dim_embedding + num_numerical, hidden_size)]
        if batch_norm:
            model += [nn.BatchNorm1d(hidden_size)]
        model += [nn.ReLU()]

        # Define blocks for hidden layers
        block = [
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        ]

        block_batch_norm = [
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        ]

        block_dropout = [
            nn.Dropout(drop),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        ]

        # Add hidden layers based on depth, batch normalization, and dropout
        for i in range(depth):
            if not batch_norm and drop == 0:
                model += block
            elif batch_norm and drop == 0:
                model += block_batch_norm
            elif drop > 0 and not batch_norm:
                model += block_dropout

        # Sequential model
        self.model = nn.Sequential(*model)

        # Output layer
        self.output = nn.Linear(hidden_size, num_classes)

    def forward(self, x_cat, x_num):
        # Convert categorical features to long type
        x_cat = x_cat.long()
        # Get embeddings for categorical features
        cat_embeddings = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        cat_embeddings = torch.stack(cat_embeddings, dim=1)  # (batch_size, num_categorical_features, dim_embedding)
        cat_embeddings = cat_embeddings.view(cat_embeddings.size(0), -1)  # Flatten to concatenate

        # Normalize numerical features
        x_num = self.numerical_norm(x_num)
        # Concatenate categorical embeddings and numerical features
        x = torch.cat([cat_embeddings, x_num], dim=1)
        # Pass through the model
        h = self.model(x)
        # Get the output
        out = self.output(h)
        return out

class TabTransformer(nn.Module):
    def __init__(self, cat_dims, num_numerical, num_classes, dim_embedding=8, num_heads=2, num_layers=2, dropout=0.1, hidden_size=None):
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
        self.embeddings = nn.ModuleList([
            nn.Embedding(cat_dim, dim_embedding) for cat_dim in cat_dims
        ])

        # Layer per le features numeriche
        self.numerical_norm = nn.LayerNorm(num_numerical) if num_numerical > 0 else None

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_embedding,
            nhead=num_heads,
            dim_feedforward=dim_embedding * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classificatore finale
        self.classifier = nn.Sequential(
            nn.Linear(len(cat_dims) * dim_embedding + (num_numerical if num_numerical > 0 else 0), hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_size, num_classes)
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
                         from_unsupervised=self.unsupervised_model,
                         weights=1)

    def predict(self, X):
        return self.network.predict(X)

    def explain(self, X):
        return self.network.explain(X)

    def feature_importances(self):
        return self.network.feature_importances_