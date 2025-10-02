import pandas as pd
import matplotlib.pyplot as plt
from utils import*
from models import*
from sklearn.metrics import r2_score

#prepare data
train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')
X_train,X_val,Y_train,Y_val=prepare_train_data(train_data) 
X_train,Y_train=prepare_train_data2(train_data)
X_test=prepare_test_data(test_data)

#correlation between features
#for elmt in X_train.columns :
#   for y in Y_train.columns :
#        correlation=train_data[elmt].corr(train_data[y])
#        print(f"Correlation entre {elmt} et {y} : {correlation:.4f}")
#print(train_data.head())
#print(test_data.head())

#calculate prediction using RF
#Y_test=random_forest(X_train,Y_train,X_test)

#caculate Y1 prediction using LR
#Y_pred1=regression(X_train,Y_train,X_val)

#calculate Y1 prediction using MLP
Y_pred1=mlp_regressor(X_train,Y_train,X_test)

#calculate prediction of Y2 using GB
#Y_pred2=gradient_boosting(X_train,Y_train,X_val) #need modification
#Y_pred=np.column_stack((Y_pred1,Y_pred2))

#calculate Y2 prediction using XGB
Y_pred2=xg_boosting(X_train,Y_train,X_test) 

#Stack  columns
Y_pred=np.column_stack((Y_pred1,Y_pred2))

preds = test_data[['id']].copy() #need modification
#preds = pd.DataFrame({
#    "Y1": Y_pred[:, 0],
#    "Y2": Y_pred[:, 1]
#})
preds['Y1'] =Y_pred[:,0]
preds['Y2'] = Y_pred[:,1]

print(preds[:10])

# Calculate R2 score
#print(f'R2 score pour Y1 : {r2_score(Y_val['Y1'],Y_pred1):.4f}')
#print(f'R2 score pour Y2 : {r2_score(Y_val['Y2'],Y_pred2):.4f}')
#print(f'R2 score : {r2_score(Y_val,Y_pred):.4f}')

#correlation= train_data['A'].corr(train_data['Y1'])
#print(f"Correlation between A and Y1 :{correlation:.4f} ")

# save preds to csv
preds.to_csv('preds.csv', index=False)