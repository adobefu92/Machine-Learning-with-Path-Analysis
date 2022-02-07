from ml_modeling import ml_modeling
from PrepareData import PrepareData
import pandas as pd
import numpy as np

def cal_marginal_effect(clf,                    # 训练好的分类模型
                        X_test,                 # 划分好的测试集数据，是一个多维数组，不需要标准化
                        col_to_var,             # 原数据集中每一列有哪些变量值，是个字典
                        col_to_var_base,        # 原数据集中每一列有哪些变量设为了base
                        continue_category,      # 连续变量（原数据列名）的集合，是个list
                        dummy_list,             # 需要虚拟的变量（原数据列名）的集合，是个list
                        X_test_column_order,    # Xtest测试集中，（Xtest数组列的order 对应哪个列名）
                        Y_list):                # Y的lab list，从小到大。
    
    marginal_var_index=[]
    marginal_var_base=[]
    marginal_var_specific=[]
    
    marignal_effets=Y_list
    
    for col_name in col_to_var.keys():
        if col_name not in continue_category:
            print(col_name)
            for var in col_to_var[col_name]:
                if var != col_to_var_base[col_name]:
                    marginal_var_index.append(col_name+"@"+str(var))
                    marginal_var_base.append(col_to_var_base[col_name])
                    marginal_var_specific.append(str(var))
    for col_name in continue_category:
        print(col_name)
        marginal_var_index.append(col_name)
        marginal_var_base.append(col_to_var_base[col_name])
        marginal_var_specific.append(str('none'))
    
    var_marginal_dataframe=pd.DataFrame("",columns=["Var base","Var specific"]+marignal_effets,index=marginal_var_index)
    var_marginal_dataframe["Var base"]=marginal_var_base
    var_marginal_dataframe["Var specific"]=marginal_var_specific
    
    base_to_Y_probability= var_marginal_dataframe.copy()
    
    ##--------------------------------------------for continuous variable 
    for col_name in continue_category:
        
        var_base=col_to_var_base[col_name]       # calculate base's probility 
        X_test_copy=X_test.copy()
        col_order=X_test_column_order[col_name];
        X_test_copy[:,col_order]=var_base
        prob_y_2 = clf.predict_proba(X_test_copy)
        base_proba=np.mean(prob_y_2, axis=0)
        
        var_marginal = [0 for x in base_proba]
        count = 0
        
        for var in col_to_var[col_name]:   
             if var != col_to_var_base[col_name]:
                 X_test_copy=X_test.copy()
                 col_order=X_test_column_order[col_name];
                 X_test_copy[:,col_order]=var
                 prob_y_2 = clf.predict_proba(X_test_copy)
                 var_proba=np.mean(prob_y_2, axis=0)
                 
                 unit_marginal = [x / (var - col_to_var_base[col_name]) for x in (var_proba-base_proba)]
                 var_marginal=[sum(x) for x in zip(var_marginal, unit_marginal)] # element-wise addition 
                 count += 1
                 
                 var_index_in_dataframe=col_name
                 #print(var_marginal)
        print(col_name, var_marginal) 
        for col in marignal_effets:            
            var_marginal_dataframe.loc[var_index_in_dataframe, col]=var_marginal[col]/count
            base_to_Y_probability.loc[var_index_in_dataframe, col]=base_proba[col]
    
    for col_name in dummy_list:
        
        X_test_copy=X_test.copy()
        for var in col_to_var[col_name]: 
            if var != col_to_var_base[col_name]:                
                col_order=X_test_column_order[col_name+"_"+str(var)]
                X_test_copy[:,col_order]=0
                prob_y_2 = clf.predict_proba(X_test_copy)
                base_proba=np.mean(prob_y_2, axis=0)   
        
       
        for var in col_to_var[col_name]: 
            
            if var != col_to_var_base[col_name]:
                X_test_for_dummy=X_test_copy.copy()      ## the value of all dummy of a variable is zeor  
                
                print(col_name,var)
                col_order=X_test_column_order[str(col_name)+"_"+str(var)]
                X_test_for_dummy[:,col_order]=1          ## change this dummy's value =1
                
                prob_y_2 = clf.predict_proba(X_test_for_dummy)
                
                var_proba=np.mean(prob_y_2, axis=0)
                 
                var_marginal=var_proba -base_proba  
                var_index_in_dataframe=col_name+"@"+str(var)
                for col in marignal_effets: 
                     var_marginal_dataframe.loc[var_index_in_dataframe, col]=var_marginal[col] 
                     base_to_Y_probability.loc[var_index_in_dataframe, col]=base_proba[col]
                     
    return var_marginal_dataframe.copy(),base_to_Y_probability.copy()



def getME(model_name,model_data,dummyVar,ConVar,DepVar,col_to_var,col_to_var_base,test_size = 0.2):

    X_train, X_test, y_train, y_test, X_test_column_order,Y_list = PrepareData(model_data,dummyVar,ConVar,DepVar,
                                                                           col_to_var_base,
                                                                           test_size)
                                                                           
                                                                           
    clf = ml_modeling(model_name,X_train, X_test, y_train, y_test)
    print(Y_list)
    
    var_marginal_dataframe, base_to_Y_probability = cal_marginal_effect(clf,
                                                X_test, 
                                                col_to_var,
                                                col_to_var_base,
                                                ConVar,
                                                dummyVar,
                                                X_test_column_order,
                                                Y_list)
    return(var_marginal_dataframe, base_to_Y_probability)                                       
                                                