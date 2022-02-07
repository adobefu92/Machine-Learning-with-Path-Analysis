import pandas as pd
from sklearn.model_selection import train_test_split

def label_coding(data,col_list):
    data_for_input_X=data[col_list].copy()
    for col_name in col_list:
        data_category= data_for_input_X[col_name].value_counts().keys().tolist()
        replace_list=list(range(0,len(data_category)))
        data_for_input_X[col_name]=data_for_input_X[col_name].replace(data_category,replace_list)
    return data_for_input_X

def delete_dummy_base(data_for_input_X,col_to_var_base):    
    base_list=[]
    for col_name in col_to_var_base.keys():
        base_list.append(col_name+"_"+str(col_to_var_base[col_name]))
        
    for col_name in data_for_input_X.columns:
        
        if col_name in base_list:
            print("delete "+ col_name)
            data_for_input_X=data_for_input_X.drop([col_name], axis=1)
    return data_for_input_X.copy()


def PrepareData(model_data,dummyVar,ConVar,DepVar,col_to_var_base,test_size = 0.2):

    data_for_input_X= pd.get_dummies(model_data[dummyVar].astype(str))
    data_for_input_X= delete_dummy_base(data_for_input_X,col_to_var_base)
    data_for_input_X[ConVar]=model_data[ConVar]
    X = data_for_input_X.values

    Y = model_data[DepVar[0]].values
    

    Y_list=model_data[DepVar[0]].value_counts().keys().tolist()
    Y_list.sort()
    X_test_column_order=dict(zip(data_for_input_X.columns.tolist(),list(range(0,len(data_for_input_X.columns.tolist())))))
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
    
    return(X_train, X_test, y_train, y_test,X_test_column_order,Y_list)