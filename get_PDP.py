from ml_modeling import ml_modeling
from PrepareData import PrepareData
import pandas as pd
import numpy as np
from sklearn.inspection import plot_partial_dependence
from sklearn.inspection import partial_dependence


def get_PDP(model_name,model_data,dummyVar,ConVar,DepVar,col_to_var_base,test_size):

    X_train, X_test, y_train, y_test, X_test_column_order,Y_list = PrepareData(model_data,dummyVar,ConVar,DepVar,
                                                                           col_to_var_base,
                                                                           test_size)
                                                                           
                                                                           
    clf = ml_modeling(model_name,X_train, X_test, y_train, y_test)
    feature = ConVar

    for i in range(len(feature)):
        feature_idx = X_test_column_order[feature[i]]
        lgb_PP_value = partial_dependence(clf,
                                          X_train,
                                          feature_idx,
                                          grid_resolution=100,kind = "average") 
        orgvalue = lgb_PP_value['values'][0]
        pred_ave = lgb_PP_value['average'][0]
        v1name = model_name + '_' + feature[i] + "_org_value"
        v2name = model_name + '_' + feature[i] + "_pred_ave"

#         print(pred_ave.shape)
#         print(orgvalue.shape)    
        df = pd.DataFrame(data = {v1name:orgvalue, v2name:pred_ave})
        if i == 0:
            partial_dependence_rs = df
        else:
            partial_dependence_rs = pd.concat([partial_dependence_rs,df],axis = 1)
            
    return(partial_dependence_rs)    
            