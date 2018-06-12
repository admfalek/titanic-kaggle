"""Titanic: Machine Learning from Disaster
Kaggle competition - simple solution"""

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

x = train_data[['SibSp']].values
y = train_data['Survived'].values

model = DummyClassifier()
model.fit(x, y)
y_pred = model.predict(x)

# prediction score
score = accuracy_score(y, y_pred)
print('score: %.2f' % score)

#test_data['Survived'] = model.predict(x)
#test_data[['PassengerId', 'Survived']].to_csv('../output/predicted-data.csv', index=False)