# %% [markdown]
# We need to install some packages (pip install "package"):
# - matplotlib
# - numpy
# - scikit-learn
# - tensorboard
# - torch

# %%
import random
import os
import time
import pickle
import itertools
import concurrent.futures


import numpy as np
import pandas as pd
from pytorch_tabular.categorical_encoders import OrdinalEncoder

from sklearn.model_selection import train_test_split, KFold
from sklearn import preprocessing

import torch
from torch import nn
from torch.nn.functional import one_hot
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter


# %%
# For reproducibility
def fix_random(seed: int) -> None:
    """Fix all the possible sources of randomness.

    Args:
        seed: the seed to use. 
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # slower

seed = 42

# %%
# Define the Data Layer
class MyDataset(Dataset):
    def __init__(self, X, y):

        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

        self.num_features = X.shape[1]
        self.num_classes = len(np.unique(y))


    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx, :], self.y[idx]

# %%
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn import preprocessing
from sklearn.decomposition import PCA

# # define a function with different normalization and scaling techniques
# def preprocess(X_train, X_val, X_test):
#
#     X_train_p, X_val_p, X_test_p = X_train, X_val, X_test
#
#     categorical_columns = X_train.select_dtypes(include=["object"]).columns.tolist()
#     print("categorical_columns len: ", len(categorical_columns))
#     numeric_columns = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
#     print("numeric_columns len: ", len(numeric_columns))
#
#     ct = ColumnTransformer(
#         [
#             ("ordinal", OneHotEncoder(handle_unknown='infrequent_if_exist', sparse_output=False), categorical_columns),  # Trasforma le colonne categoriche
#             ("scale", StandardScaler(), numeric_columns)  # Normalizza le colonne numeriche
#         ],
#         remainder="passthrough"  # Mantieni le altre colonne invariate
#     )
#     ct = ct.fit(X_train)
#     ct = ct.set_output(transform="pandas")
#
#     X_train_p = ct.transform(X_train_p)
#     X_val_p = ct.transform(X_val_p)
#     X_test_p = ct.transform(X_test_p)
#
#     columns = X_train_p.columns.tolist()
#
#     # print("columns: ", columns)
#     one_hot_columns = []
#     for col in columns:
#         if col.startswith("ordinal__"):
#             one_hot_columns.append(col)
#
#     # print("one_hot_columns: ", one_hot_columns)
#
#     ct2 = ColumnTransformer(
#         [
#             ("ordinal", PCA(n_components=0.95), one_hot_columns),  # Trasforma le colonne categoriche
#         ],
#         remainder="passthrough"  # Mantieni le altre colonne invariate
#     )
#     ct2 = ct2.fit(X_train_p)
#     X_train_p = ct2.transform(X_train_p)
#     X_val_p = ct2.transform(X_val_p)
#     X_test_p = ct2.transform(X_test_p)
#
#     # pca = PCA(n_components=0.9999).fit(X_train_p)
#     #
#     # print(pca.explained_variance_ratio_)
#     # print("Varianza cumulativa:", pca.explained_variance_ratio_.cumsum())
#     #
#     # print("Forma originale:", X_train_p.shape)
#     # X_train_p = pca.transform(X_train_p)
#     # print("Forma dopo PCA:", X_train_p.shape)
#     # X_val_p = pca.transform(X_val_p)
#     # X_test_p = pca.transform(X_test_p)
#
#
#     return X_train_p, X_val_p, X_test_p

# define a function with different normalization and scaling techniques
def preprocess(X_train, X_val):

    X_train_p, X_val_p = X_train, X_val

    categorical_columns = X_train.select_dtypes(include=["object"]).columns.tolist()
    print("categorical_columns len: ", len(categorical_columns))
    numeric_columns = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    print("numeric_columns len: ", len(numeric_columns))

    ct = ColumnTransformer(
        [
            ("ordinal", OneHotEncoder(handle_unknown='infrequent_if_exist', sparse_output=False), categorical_columns),  # Trasforma le colonne categoriche
            ("scale", StandardScaler(), numeric_columns)  # Normalizza le colonne numeriche
        ],
        remainder="passthrough"  # Mantieni le altre colonne invariate
    )
    ct = ct.fit(X_train)
    # ct = ct.set_output(transform="pandas")

    X_train_p = ct.transform(X_train_p)
    X_val_p = ct.transform(X_val_p)

    # columns = X_train_p.columns.tolist()

    # # print("columns: ", columns)
    # one_hot_columns = []
    # for col in columns:
    #     if col.startswith("ordinal__"):
    #         one_hot_columns.append(col)

    # # print("one_hot_columns: ", one_hot_columns)

    # ct2 = ColumnTransformer(
    #     [
    #         ("ordinal", PCA(n_components=0.95), one_hot_columns),  # Trasforma le colonne categoriche
    #     ],
    #     remainder="passthrough"  # Mantieni le altre colonne invariate
    # )
    # ct2 = ct2.fit(X_train_p)
    # X_train_p = ct2.transform(X_train_p)
    # X_val_p = ct2.transform(X_val_p)


    return X_train_p, X_val_p

# %%
# Architecture

class FeedForwardPlus(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size, depth=1, batch_norm=False, drop=0):
        super(FeedForwardPlus, self).__init__()

        model = []
        model += [nn.Linear(input_size, hidden_size)]
        if batch_norm:
            model += [nn.BatchNorm1d(hidden_size)]
        model += [nn.ReLU()]

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

        for i in range(depth):
            if not batch_norm and drop == 0:
                model += block
            elif batch_norm and drop == 0:
                model += block_batch_norm
            elif drop > 0 and not batch_norm:
                model += block_dropout
        
        self.model = nn.Sequential(*model)
        
        self.output = nn.Linear(hidden_size, num_classes)
        

    def forward(self, x):
        h = self.model(x)
        out = self.output(h)
        return out

# %%
# Define a function for the training process

def train_model(model: FeedForwardPlus, criterion, optimizer, epoch, scheduler, train_loader, val_loader, device, writer, log_name="model"):
    n_iter = 0
    best_valid_loss = float('inf')
    for epoch in range(epoch):
        model.train()
        
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()

            # Forward pass
            y_pred = model(data)

            # Compute Loss
            loss = criterion(y_pred, targets)
            writer.add_scalar("Loss/train", loss, n_iter)

            # Backward pass
            loss.backward()
            optimizer.step()

            n_iter += 1
        
        labels, _, y_pred = test_model(model, val_loader, device)
        loss_val = criterion(y_pred, labels)
        writer.add_scalar("Loss/val", loss_val, epoch)

        # print("Epoch:", epoch, "Loss:", loss_val.item())


        # save best model
        if loss_val.item() < best_valid_loss:
            best_valid_loss = loss_val.item()
            if not os.path.exists(f'{filepath}/models'):
                os.makedirs(f'{filepath}/models')
            # torch.save(model.state_dict(), f'{filepath}/models/'+log_name)
        
        writer.add_scalar("hparam/Learning Rate", scheduler.get_last_lr()[0], epoch)
        
        scheduler.step()
            
    return model, best_valid_loss

# %%
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


# Define a function to evaluate the performance on validation and test sets

def test_model(model, data_loader, device):
    model.eval()
    y_pred = []
    y_test = []

    for data, targets in data_loader:
        data, targets = data.to(device), targets.to(device)
        y_pred += model(data)
        #print(y_pred)
        y_test += targets
        #print(targets)

    y_test = torch.stack(y_test).squeeze()
    y_pred = torch.stack(y_pred).squeeze()
    y_pred_c = y_pred.argmax(dim=1, keepdim=True).squeeze()

    return y_test, y_pred_c, y_pred

def evaluate_model(y_test, y_pred ):
    acc = ((y_test == y_pred).float().sum() / y_test.shape[0]).cpu().numpy()
    bacc = balanced_accuracy_score(y_test.cpu().numpy(), y_pred.cpu().numpy())
    f1 = f1_score(y_test.cpu().numpy(), y_pred.cpu().numpy(), average="weighted")

    perf = {"acc": acc, "bacc": bacc, "f1": f1}

    return perf

def train_and_evaluate(batch_size, hidden_size, depth, gamma, batch_norm, X, y, kf, seed, num_epochs, learning_rate, step_size, device):
    fix_random(seed)
    start = time.time()

    log_name = f"B{batch_size}-dim{hidden_size}-dp{depth}-ep{num_epochs}-lr{learning_rate}-steplr{step_size}-gamma{gamma}-BN{batch_norm}-drop{drop}"
    print(log_name)

    # start tensorboard
    writer = SummaryWriter(f'{filepath}/runs/{log_name}')
    accuracy_per_fold = []
    balanced_accuracy_score_per_fold = []
    f1_score_per_fold = []
    best_loss_per_fold = []

    fold = 1
    for train_index, val_index in kf.split(X, y):
        print(f"{log_name}: Fold {fold}")
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y[train_index], y[val_index]

        X_train, X_val = preprocess(X_train, X_val)

        train_dataset = MyDataset(X_train, y_train)
        val_dataset = MyDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # define architecture, loss and optimizer
        model = FeedForwardPlus(train_dataset.num_features, train_dataset.num_classes, hidden_size, depth, batch_norm=batch_norm)
        model.to(device)

        # train
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        model, best_valid_loss = train_model(model, criterion, optimizer, num_epochs, scheduler, train_loader, val_loader, device, writer, log_name)

        # Valuta il modello sul validation set
        y_test, y_pred_c, _ = test_model(model, val_loader, device)
        perf = evaluate_model(y_test, y_pred_c)

        print(f"{log_name}: Fold {fold} - Accuracy: {perf['acc']:.2f}%")
        print(f"{log_name}: Fold {fold} - Balanced Accuracy: {perf['bacc']:.2f}%")
        print(f"{log_name}: Fold {fold} - F1 Score: {perf['f1']:.2f}%")

        accuracy_per_fold.append(perf["acc"])
        balanced_accuracy_score_per_fold.append(perf["bacc"])
        f1_score_per_fold.append(perf["f1"])
        best_loss_per_fold.append(best_valid_loss)
        fold += 1

    metrics_dict = {
            'best loss': np.mean(best_loss_per_fold),
            'avg accuracy': np.mean(accuracy_per_fold),
            'std accuracy': np.std(accuracy_per_fold),
            'avg balanced accuracy': np.mean(balanced_accuracy_score_per_fold),
            'std balanced accuracy': np.std(balanced_accuracy_score_per_fold),
            'avg f1 score': np.mean(f1_score_per_fold),
            'std f1 score': np.std(f1_score_per_fold)
    }

    # Riassunto dei risultati
    print("{log_name}: Cross-Validation Results:")
    print(f"Average Accuracy: {metrics_dict['avg accuracy']*100:.2f}%")
    print(f"Standard Deviation of Accuracy: {metrics_dict['std accuracy']*100:.2f}%")
    print(f"Average Balanced Accuracy: {metrics_dict['avg balanced accuracy']*100:.2f}%")
    print(f"Standard Deviation of Balanced Accuracy: {metrics_dict['std balanced accuracy']*100:.2f}%")
    print(f"Average F1 Score: {metrics_dict['avg f1 score']*100:.2f}%")
    print(f"Standard Deviation of F1 Score: {metrics_dict['std f1 score']*100:.2f}%")

    # Log hyperparameters and metrics to TensorBoard
    writer.add_hparams(
        {
            'hparam/bsize': batch_size,
            'hparam/hidden size': hidden_size,
            'hparam/depth': depth + 2,
            'hparam/scheduler': gamma,
            'hparam/batch norm': batch_norm
        },
        metrics_dict
    )

    # Dopo il ciclo su tutti gli hyperparameters
    writer.add_scalars("Metrics/Accuracy", {
        f"Run_{log_name}": metrics_dict['avg accuracy']
    }, global_step=0)

    writer.add_scalars("Metrics/Balanced Accuracy", {
        f"Run_{log_name}": metrics_dict['avg balanced accuracy']
    }, global_step=0)

    writer.add_scalars("Metrics/F1 Score", {
        f"Run_{log_name}": metrics_dict['avg f1 score']
    }, global_step=0)

    writer.flush()
    print(f"Log: {log_name}")
    print(f"Best loss: {min(best_loss_per_fold)}")
    print(f"Time elapsed: {time.time() - start}")

    return log_name, metrics_dict

if __name__ == "__main__":
    # %%
    # look for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('mps')
    print("Device: {}".format(device))

    # %%
    from sklearn.model_selection import StratifiedKFold

    save_in_test_folder = False
    if save_in_test_folder:
        filepath = "./TestModule"
    else:
        filepath = "./TrainingModule"


    FILENAME = "TrainingModule/dataset/train_dataset.csv"

    #Prepare train data
    data = pd.read_csv(FILENAME, sep=",", low_memory=False)

    # get features names
    features = list(data.columns)
    features_to_remove = ["label", "ts", "src_ip", "dst_ip", "dns_query", "ssl_subject", "ssl_issuer", "http_uri", "type", "http_referrer", "http_user_agent"]
    features = [feature for feature in features if feature not in features_to_remove]
    data = data[features + ["type"]]

    # Converte i valori in numeri, sostituendo quelli non validi con NaN
    data["src_bytes"] = pd.to_numeric(data["src_bytes"], errors='coerce')
    # Filtra le righe con NaN (valori non convertibili)
    data = data.dropna(subset=["src_bytes"])
    # Converte i valori rimasti in interi
    data.loc[:, "src_bytes"] = data["src_bytes"].astype(int)

    print("#Righe: " + str(data.shape[0]) + " #Colonne: " + str(data.shape[1]))
    df1 = data.dropna()
    print("#Righe: " + str(df1.shape[0]) + " #Colonne: " + str(data.shape[1]))


    # data = data.sample(n=1000, random_state=5)

    X = data[features]
    y = data["type"]

    le = preprocessing.LabelEncoder()
    le.fit(y)
    with open(f"{filepath}/transformer/target_encoder.save", "wb") as f:
        pickle.dump(le, f)

    y = le.transform(y)

    # Separate indices
    indices = np.arange(X.shape[0])
    train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=y, random_state=seed)

    X_val = X.iloc[val_idx]
    y_val = y[val_idx]
    X_train = X.iloc[train_idx]
    y_train = y[train_idx]

    # #Prepare test data
    # df2 = pd.read_csv("TrainingModule/dataset/test_dataset.csv", sep=",", low_memory=False)
    # df2 = df2.dropna()
    # X_test = df2[features]
    # y_test = df2["type"].to_numpy()
    # y_test = le.transform(y_test)


    # X_train, X_val, X_test = preprocess(X_train, X_val, X_test)
    X_train, X_val = preprocess(X_train, X_val)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    # print(X_test.shape[0])

    # Create the dataset
    train_dataset = MyDataset(X_train,y_train)
    val_dataset = MyDataset(X_val,y_val)
    # test_dataset = MyDataset(X_test,y_test)

   
    # Hyperparameters
    # seed = 42
    batch_sizes = [16, 32] #64
    hidden_sizes = [32, 64, 128] # 64
    batch_norm_list = [False, True]
    drop = 0
    depths = [2, 4]
    num_epochs = 50
    learning_rate = 0.01
    gammas = [0.9, 0.5]
    step_size = num_epochs / 4

    # Set up the hyperparameters to test
    hyperparameters = list(itertools.product(batch_sizes, hidden_sizes, depths, gammas, batch_norm_list))


    results = {}
    for batch_size, hidden_size, depth, gamma, batch_norm in hyperparameters[20:24]:
        #  Cross-validation
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        log_name, metrics_dict = train_and_evaluate(batch_size, hidden_size, depth, gamma, batch_norm, X, y, kf, seed, num_epochs, learning_rate, step_size, device)    
        results[log_name] = metrics_dict

    print(results)
        # print(f"Finished {log_name} with Average Accuracy: {avg_accuracy:.2f}% and Std Accuracy: {std_accuracy:.2f}%")
    # Use ThreadPoolExecutor or ProcessPoolExecutor depending on your needs (ProcessPoolExecutor for CPU-bound tasks)
    # with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
    #     try:
    #         futures = []
    #         for batch_size, hidden_size, depth, gamma, batch_norm in hyperparameters:
    #             futures.append(executor.submit(train_and_evaluate, batch_size, hidden_size, depth, gamma, batch_norm, X, y, kf, seed, num_epochs, learning_rate, step_size, device))

    #         # Wait for all tasks to complete
    #         for future in concurrent.futures.as_completed(futures):
    #             log_name, avg_accuracy, std_accuracy = future.result()
    #             print(f"Finished {log_name} with Average Accuracy: {avg_accuracy:.2f}% and Std Accuracy: {std_accuracy:.2f}%")
    #     except Exception as e:
    #         print(f"Error occurred during training")

# %% [markdown]
# Run Tensorboard from the command line:
# 
# "tensorboard --logdir runs/"

# # %%
# start = time.time()
# # B32-dim32-dp4-ep100-lr0.01-steplr25.0-gamma0.5-BNFalse-drop0
# # hyperparatemeters
# batch_size = 32
# depth = 4
# hidden_size = 32
# batch_norm = False
# drop = 0
# num_epochs = 50  # try 100, 200, 500
# learning_rate = 0.01
# gamma=0.5
# step_size=num_epochs/4


# log_name = "B"+str(batch_size)+"-dim"+str(hidden_size)+"-dp"+str(depth)+"-ep"+str(num_epochs)+"-lr"+str(learning_rate)+"-steplr"+str(step_size)+"-gamma"+str(gamma)+"-BN"+str(batch_norm)+"-drop"+str(drop)+"prova"

# # fix the seed for reproducibility
# fix_random(seed)


# # Create relative dataloaders
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size)
# # test_loader = DataLoader(test_dataset, batch_size=batch_size)


# # Define the architecture, loss and optimizer
# model = FeedForwardPlus(train_dataset.num_features, train_dataset.num_classes, hidden_size, depth, batch_norm=batch_norm, drop=drop)
# print(model)
# model.to(device)

# # Define the training elements
# criterion = torch.nn.CrossEntropyLoss()
# # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# # Test before the training
# y_test, y_pred_c, _ = test_model(model, val_loader, device)
# perf = evaluate_model(y_test, y_pred_c)

# print(f"Before training Accuracy: {perf['acc']:.2f}%")
# print(f"Before training Balanced Accuracy: {perf['bacc']:.2f}%")
# print(f"Before training F1 Score: {perf['f1']:.2f}%")

# # Start tensorboard
# writer = SummaryWriter(f'{filepath}/runs/'+log_name)


# # Train the model
# model, best_valid_loss = train_model(model, criterion, optimizer, num_epochs, scheduler, train_loader, val_loader, device, writer, log_name)


# # Load best model
# model.load_state_dict(torch.load(f"{filepath}/models/"+log_name, weights_only=True))
# model.to(device)


# # Test after the training
# y_test, y_pred_c, _ = test_model(model, val_loader, device)
# perf = evaluate_model(y_test, y_pred_c)

# print(f"After training Accuracy: {perf['acc']:.2f}%")
# print(f"After training Balanced Accuracy: {perf['bacc']:.2f}%")
# print(f"After training F1 Score: {perf['f1']:.2f}%")


# # Close tensorboard writer after a training
# writer.flush()
# writer.close()

# # Save timestamp
# end = time.time()
# print("Time elapsed:", end - start)







# # %%

# # Cross-validation
# kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# # Grid Search

# # Hyperparameters
# seed = 42
# batch_sizes = [16, 32, 64]
# hidden_sizes = [32, 64, 128] # 64
# batch_norm_list = [False, True]
# drop = 0
# depths = [2, 4]
# num_epochs = 50
# learning_rate = 0.01
# gammas = [1, 0.5]
# step_size = num_epochs / 4

# import itertools
# hyperparameters = itertools.product(batch_sizes, hidden_sizes, depths, gammas, batch_norm_list)

# #grid search loop
# for batch_size, hidden_size, depth, gamma, batch_norm in hyperparameters:
#     fix_random(seed)
#     start = time.time()

#     log_name = "B"+str(batch_size)+"-dim"+str(hidden_size)+"-dp"+str(depth)+"-ep"+str(num_epochs)+"-lr"+str(learning_rate)+"-steplr"+str(step_size)+"-gamma"+str(gamma)+"-BN"+str(batch_norm)+"-drop"+str(drop)
#     print(log_name, end=", ")

#     #start tensorboard
#     writer = SummaryWriter('runs/'+log_name)
#     accuracy_per_fold = []
#     balanced_accuracy_score_per_fold = []
#     f1_score_per_fold = []
#     best_loss_per_fold = []

#     fold = 1
#     for train_index, val_index in kf.split(X, y):
#         print(f"Fold {fold}")
#         X_train, X_val = X.iloc[train_index], X.iloc[val_index]
#         y_train, y_val = y[train_index], y[val_index]

#         # X_train, X_val, X_test = preprocess(X_train, X_val, X_test)
#         X_train, X_val = preprocess(X_train, X_val)

#         train_dataset = MyDataset(X_train, y_train)
#         val_dataset = MyDataset(X_val, y_val)
#         # test_dataset = MyDataset(X_test, y_test)

#         # Create relative dataloaders
#         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#         val_loader = DataLoader(val_dataset, batch_size=batch_size)

#         #define architecture, loss and optimizer
#         model = FeedForwardPlus(train_dataset.num_features, train_dataset.num_classes, hidden_size, depth, batch_norm=batch_norm)
#         model.to(device)

#         # train
#         criterion = torch.nn.CrossEntropyLoss()
#         # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
#         optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
#         model, best_valid_loss = train_model(model, criterion, optimizer, num_epochs, scheduler, train_loader, val_loader, device, writer, log_name)

#         # Valuta il modello sul validation set
#         y_test, y_pred_c, _ = test_model(model, val_loader, device)
#         perf = evaluate_model(y_test, y_pred_c)

#         print(f"Fold {fold} - Accuracy: {perf["acc"]:.2f}%")
#         print(f"Fold {fold} - Balanced Accuracy: {perf["bacc"]:.2f}%")
#         print(f"Fold {fold} - F1 Score: {perf["f1"]:.2f}%")

#         accuracy_per_fold.append(perf["acc"])
#         balanced_accuracy_score_per_fold.append(perf["bacc"])
#         f1_score_per_fold.append(perf["f1"])
#         best_loss_per_fold.append(best_valid_loss)
#         fold += 1

#     # Riassunto dei risultati
#     print("Cross-Validation Results:")
#     print(f"Average Accuracy: {np.mean(accuracy_per_fold)*100:.2f}%")
#     print(f"Standard Deviation of Accuracy: {np.std(accuracy_per_fold)*100:.2f}%")
#     print(f"Average Balanced Accuracy: {np.mean(balanced_accuracy_score_per_fold)*100:.2f}%")
#     print(f"Standard Deviation of Balanced Accuracy: {np.std(balanced_accuracy_score_per_fold)*100:.2f}%")
#     print(f"Average F1 Score: {np.mean(f1_score_per_fold)*100:.2f}%")
#     print(f"Standard Deviation of F1 Score: {np.std(f1_score_per_fold)*100:.2f}%")


#     # Close tensorboard writer after a training
#     # Log hyperparameters and metrics to TensorBoard
#     writer.add_hparams(
#         {
#             'hparam/bsize': batch_size,
#             'hparam/hidden size': hidden_size,
#             'hparam/depth': depth + 2,
#             'hparam/scheduler': gamma,
#             'hparam/batch norm': batch_norm
#         },
#         {
#             'best loss': np.mean(best_loss_per_fold),
#             'avg accuracy': np.mean(accuracy_per_fold),
#             'std accuracy': np.std(accuracy_per_fold),
#             'avg balanced accuracy': np.mean(balanced_accuracy_score_per_fold),
#             'std balanced accuracy': np.std(balanced_accuracy_score_per_fold),
#             'avg f1 score': np.mean(f1_score_per_fold),
#             'std f1 score': np.std(f1_score_per_fold)
#         }
#     )
#     writer.flush()
#     print("best loss:", best_valid_loss)
#     print("time elapsed:", time.time() - start)
# writer.close()

# %%
# Choose and load the best model and evaluate it on the test set
# filename = "models/B16-dim16-dp4-ep100-lr0.01-steplr25.0-gamma1-BNFalse-drop0"

# get all files in the models folder
# files = os.listdir('models')

# for filename in files:
#     if not filename.endswith('drop0'):
#         continue
#     print(filename)
#     filename_list = filename.split('-')

#     batch_size = int(filename_list[0].split('B')[1])
#     hidden_size = int(filename_list[1].split('dim')[1])
#     depth = int(filename_list[2].split('dp')[1])

#     model = FeedForwardPlus(train_dataset.num_features, train_dataset.num_classes, hidden_size, depth)
#     state_dict = torch.load('models/' + filename, map_location=device)

#     # Remove unexpected keys from state_dict
#     model_state_dict = model.state_dict()
#     for key in list(state_dict.keys()):
#         if key not in model_state_dict or state_dict[key].shape != model_state_dict[key].shape:
#             del state_dict[key]

#     model.load_state_dict(state_dict, strict=False)
#     model.to(device)

#     y_test, y_pred_c, _ = test_model(model, test_loader, device)
#     acc = (y_test == y_pred_c).float().sum() / y_test.shape[0]

#     print('Accuracy of the best model on the test set:', acc.cpu().numpy())

# %% [markdown]
# B16-dim16-dp1-ep100-lr0.01-steplr25-gamma0.5-BNTrue-drop0
# Accuracy of the best model on the test set: 0.013589648
# B16-dim16-dp2-ep100-lr0.01-steplr25-gamma0.5-BNTrue-drop0
# Accuracy of the best model on the test set: 0.07705539
# B16-dim16-dp2-ep100-lr0.01-steplr25.0-gamma0.5-BNFalse-drop0
# Accuracy of the best model on the test set: 0.7052449
# B16-dim16-dp2-ep100-lr0.01-steplr25.0-gamma0.5-BNTrue-drop0
# Accuracy of the best model on the test set: 0.03152912
# B16-dim16-dp2-ep100-lr0.01-steplr25.0-gamma1-BNFalse-drop0
# Accuracy of the best model on the test set: 0.70811635
# B16-dim16-dp2-ep100-lr0.01-steplr25.0-gamma1-BNTrue-drop0
# Accuracy of the best model on the test set: 0.23312311
# B16-dim16-dp4-ep100-lr0.01-steplr25.0-gamma0.5-BNFalse-drop0
# Accuracy of the best model on the test set: 0.7009709
# B16-dim16-dp4-ep100-lr0.01-steplr25.0-gamma0.5-BNTrue-drop0
# Accuracy of the best model on the test set: 0.25875294
# B16-dim16-dp4-ep100-lr0.01-steplr25.0-gamma1-BNFalse-drop0
# Accuracy of the best model on the test set: 0.50005925
# B16-dim16-dp4-ep100-lr0.01-steplr25.0-gamma1-BNTrue-drop0
# Accuracy of the best model on the test set: 0.25875294
# B16-dim32-dp2-ep100-lr0.01-steplr25.0-gamma0.5-BNFalse-drop0
# Accuracy of the best model on the test set: 0.7323294
# B16-dim32-dp2-ep100-lr0.01-steplr25.0-gamma0.5-BNTrue-drop0
# Accuracy of the best model on the test set: 0.094691604
# B16-dim32-dp2-ep100-lr0.01-steplr25.0-gamma1-BNFalse-drop0
# Accuracy of the best model on the test set: 0.6785584
# B16-dim32-dp2-ep100-lr0.01-steplr25.0-gamma1-BNTrue-drop0
# Accuracy of the best model on the test set: 0.08437143
# B16-dim32-dp4-ep100-lr0.01-steplr25.0-gamma0.5-BNFalse-drop0
# Accuracy of the best model on the test set: 0.6625806
# B16-dim32-dp4-ep100-lr0.01-steplr25.0-gamma0.5-BNTrue-drop0
# Accuracy of the best model on the test set: 0.19223097
# B16-dim32-dp4-ep100-lr0.01-steplr25.0-gamma1-BNFalse-drop0
# Accuracy of the best model on the test set: 0.46978578
# B16-dim32-dp4-ep100-lr0.01-steplr25.0-gamma1-BNTrue-drop0
# Accuracy of the best model on the test set: 0.24521069
# B32-dim16-dp2-ep100-lr0.01-steplr25.0-gamma0.5-BNFalse-drop0
# Accuracy of the best model on the test set: 0.739418
# B32-dim16-dp2-ep100-lr0.01-steplr25.0-gamma0.5-BNTrue-drop0
# Accuracy of the best model on the test set: 0.10031131
# B32-dim16-dp2-ep100-lr0.01-steplr25.0-gamma1-BNFalse-drop0
# Accuracy of the best model on the test set: 0.6307956
# B32-dim16-dp2-ep100-lr0.01-steplr25.0-gamma1-BNTrue-drop0
# Accuracy of the best model on the test set: 0.12023142
# B32-dim16-dp4-ep100-lr0.01-steplr25.0-gamma0.5-BNFalse-drop0
# Accuracy of the best model on the test set: 0.74385786
# B32-dim16-dp4-ep100-lr0.01-steplr25.0-gamma0.5-BNTrue-drop0
# Accuracy of the best model on the test set: 0.24466105
# B32-dim16-dp4-ep100-lr0.01-steplr25.0-gamma1-BNFalse-drop0
# Accuracy of the best model on the test set: 0.6837232
# B32-dim16-dp4-ep100-lr0.01-steplr25.0-gamma1-BNTrue-drop0
# Accuracy of the best model on the test set: 0.116355434
# B32-dim32-dp2-ep100-lr0.01-steplr25.0-gamma0.5-BNFalse-drop0
# Accuracy of the best model on the test set: 0.7603142
# B32-dim32-dp2-ep100-lr0.01-steplr25.0-gamma0.5-BNTrue-drop0
# Accuracy of the best model on the test set: 0.11939273
# B32-dim32-dp2-ep100-lr0.01-steplr25.0-gamma1-BNFalse-drop0
# Accuracy of the best model on the test set: 0.7040603
# B32-dim32-dp2-ep100-lr0.01-steplr25.0-gamma1-BNTrue-drop0
# Accuracy of the best model on the test set: 0.1295897
# B32-dim32-dp4-ep100-lr0.01-steplr25.0-gamma0.5-BNFalse-drop0
# Accuracy of the best model on the test set: 0.789768
# B32-dim32-dp4-ep100-lr0.01-steplr25.0-gamma0.5-BNTrue-drop0
# Accuracy of the best model on the test set: 0.13042839
# B32-dim32-dp4-ep100-lr0.01-steplr25.0-gamma1-BNFalse-drop0
# Accuracy of the best model on the test set: 0.68777454
# B32-dim32-dp4-ep100-lr0.01-steplr25.0-gamma1-BNTrue-drop0
# Accuracy of the best model on the test set: 0.13042839


