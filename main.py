import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import pickle

df = pd.read_csv('Salary.csv')
# df.head()

X = df.iloc[:, :-1].values   
y = df.iloc[:, -1].values 


# random_state => seed value used by random number generator
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

# sns.displot(predictions-y_test)

plt.scatter(X_train, y_train, color='red')
# plt.plot(X_train, model.predict(X_train))

pickle.dump(model,open('model.pkl','wb'))
