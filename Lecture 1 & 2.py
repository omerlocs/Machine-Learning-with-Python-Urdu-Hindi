from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
data = pd.read_csv('train.csv')
y = data.SalePrice
n_data = ['MSSubClass','LotArea']
X= data[n_data]
fea_train,fea_label,label_train,label_fea = train_test_split(X,y,random_state=1)
clf = GaussianNB()
clf.fit(fea_train,label_train)
predict = clf.predict(fea_label)
print(predict)
print(accuracy_score(predict,label_fea)*100)

