from sklearn.model_selection import train_test_split

def prepare_train_data(train_data):
    X_tr=train_data[['A','B','C','D','E','F','G','H','I','J','K','L','M','N']]
    Y_tr=train_data[['Y1','Y2']]
    X_train,X_val,Y_train,Y_val=train_test_split(X_tr,Y_tr,test_size=0.2,random_state=42)
    return X_train,X_val,Y_train,Y_val

def prepare_train_data2(train_data):
    X_train=train_data[['A','B','C','D','E','F','G','H','I','J','K','L','M','N']]
    Y_train=train_data[['Y1','Y2']]
    return X_train,Y_train

def prepare_test_data(test_data):
    X_test=test_data[['A','B','C','D','E','F','G','H','I','J','K','L','M','N']]
    return X_test
