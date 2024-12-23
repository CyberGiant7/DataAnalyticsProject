MY_UNIQUE_ID = "TestUser"

# Output: unique ID of the team
def getName():
    return MY_UNIQUE_ID

# Input: Test dataframe
# Output: PreProcessed test dataframe
def preprocess(data, clfName):
    pass

# Input: Classifier name ("lr": Logistic Regression, "svc": Support Vector Classifier)
# Output: Classifier object
def load(clfName):
    pass

# Input: PreProcessed dataset, Classifier Name, Classifier Object
# Output: Performance dictionary
def predict(data, clfName, clf):
    pass