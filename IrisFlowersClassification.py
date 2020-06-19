import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


def getInfo(dataset, distributionGroup):
    print('-'*20)
    print('Shape:', dataset.shape)
    print('First 20 Rows:')
    print(dataset.head(20))
    print('Quick Summary:')
    print(dataset.describe())
    print('Class distribution:')
    print(dataset.groupby(distributionGroup).size())
    print('-'*20)

def visualize(dataset):
    dataset.plot(kind='box', subplots = True, layout = (2,2), sharex = False, sharey = False)
    plt.show()

    dataset.hist()
    plt.show()

    scatter_matrix(dataset)
    plt.show()

def visualizeAlgorithmComparison(results, names):
    plt.boxplot(results, labels = names)
    plt.title('Algorithm Comparison')
    plt.show()

# Load in dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names = names)

# Brief data visualization
getInfo(dataset, 'class')
visualize(dataset)

# Validation Dataset (80% train, 20% validate)
arr = dataset.values    # Numpy representation of dataframe
x = arr[:, 0:4]         # Calculations from dataset
y = arr[:, 4]           # Classes of flowers
x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size = 0.2, random_state = 1)

# Set up various ML Models
models = []
models.append(('LR', LogisticRegression(solver = 'liblinear', multi_class = 'ovr')))    # Logistic Regression
models.append(('LDA', LinearDiscriminantAnalysis()))                                    # Linear Discriminant Analysis
models.append(('KNN', KNeighborsClassifier()))                                          # K-Nearest Neighbors
models.append(('CART', DecisionTreeClassifier()))                                         # Classification and Regression Trees
models.append(('NB', GaussianNB()))                                                     # Gaussian Naive Bayes
models.append(('SVM', SVC(gamma = 'auto')))                                             # Support Vector Machine

# Evaluate results with each model
results = []
names = []
for name, model in models:
    # Use stratified 10-fold cross-validation to estimate model accuracy
    kfold = StratifiedKFold(n_splits = 10, random_state = 1, shuffle = True)
    cv_results = cross_val_score(model, x_train, y_train, cv = kfold, scoring = 'accuracy') # accuracy = correctly predicted / total instances
    results.append(cv_results)
    names.append(name)
    print(f'{name}\n~~~~\nMean: {cv_results.mean()}\nStandard Deviation: {cv_results.std()}')
    print('-'*50)

visualizeAlgorithmComparison(results, names)    # Helps to visualize just how accurate the support vector machine is

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Based on the above results, we will proceed with utilizing a support vector machine for our final model.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# Now let's test the model against the validation set (to ensure quality of training results)
model = SVC(gamma = 'auto')
model.fit(x_train, y_train)
predictions = model.predict(x_validation)   # Attempt prediction of flower class based on length/width statistics

# Evaluate predictions
print('Accuracy Score:', accuracy_score(y_validation, predictions)) # Accuracy on validation/hold out dataset
print('Confusion Matrix:')                                          # An indication of the three errors made
print(confusion_matrix(y_validation, predictions))
print('Classification Report:')                                     # Breakdown of each class's prediction statistics
print(classification_report(y_validation, predictions))