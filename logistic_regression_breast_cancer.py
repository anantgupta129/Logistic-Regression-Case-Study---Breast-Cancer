# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Logistic Regresison Brest Cancer Case study

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# %%
dts = pd.read_csv('breast_cancer.csv')
X = dts.iloc[:, 1:-1].values
y = dts.iloc[:, -1].values


# %%
y = np.reshape(y, (-1,1))

# %% [markdown]
# ## Splitting dataset into training and test set

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)

# %% [markdown]
# ## Training the logistic regression model on Training set

# %%
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0)
clf.fit(X_train, y_train)

# %% [markdown]
# ## Predicting tet set result

# %%
y_pred = clf.predict(X_test)


# %%
print(y_pred)

# %% [markdown]
# ## computing Confusion Matrix and accuracy

# %%
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accu = accuracy_score(y_test, y_pred)
print("Test set Accuracy: {:.3f}%".format(accu*100))

# %% [markdown]
# ## Computing the accuracy with k-Fold Cross Validation

# %%
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=clf, cv=10, X=X_train, y=y_train)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard deviation: {:.2f} %".format(accuracies.std()*100))

# %% [markdown]
# ## Applying Grid Search to find the best model and the best parameters

# %%
from sklearn.model_selection import GridSearchCV
parameters = [{'penalty': ['l1', 'l2', 'elasticnet'], 'C':[0.25, 0.5, 0.75, 1], 'random_state': [0, 42, 50, 69, 129], 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}]
grid_search = GridSearchCV(estimator=clf, 
                           param_grid=parameters, 
                           scoring='accuracy', 
                           cv=10, 
                           n_jobs=-1)
grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.3f}%".format(best_accuracy*100))
print("Best Parameters:", best_parameters)

# %% [markdown]
# ## Linear Regression Model with best parameters 

# %%
from sklearn.linear_model import LogisticRegression
clf_2 = LogisticRegression(random_state=0, C=0.25, penalty='l1', solver='saga')
clf_2.fit(X_train, y_train)


# %%
y_pred_2 = clf_2.predict(X_test)
print(y_pred_2)


# %%
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=clf_2, cv=10, X=X_train, y=y_train)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard deviation: {:.2f} %".format(accuracies.std()*100))

