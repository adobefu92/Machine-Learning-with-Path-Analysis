import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from getME import getME


def loop_all_model(list_of_models,
                   model_data,
                   RegOrdVar,
                   OrdCatVar,
                   ConVar,
                   FirstDepVar,
                   SecondDepVar,
                   col_to_var,
                   col_to_var_base,
                   test_size = 0.2):

    for model_name in list_of_models:
        
        dummyVar = RegOrdVar + OrdCatVar + FirstDepVar
        DepVar = SecondDepVar
        
        var_marginal_dataframe, base_to_Y_probability =getME(model_name,model_data,
                                                              dummyVar,ConVar,DepVar,
                                                              col_to_var,col_to_var_base,
                                                              test_size = test_size)
        # Var base --> Y2 base value name; Var specific --> Y2 non-base value name
        var_marginal_dataframe.columns=["Var base","Var specific",
                                'car','other_modes']
        # Var base --> U- Y2 base value name; Var specific --> U- Y2 non-base value name
        base_to_Y_probability.columns=["Var base","Var specific",
                                       "U-"+'car',"U-"+'other_modes']
        direct_impacts_on_Y2=pd.concat([var_marginal_dataframe, base_to_Y_probability ],axis=1)

        direct_impacts_on_Y2.to_csv(model_name+"direct_impacts_on_Y2"+ ".csv")
        
        dummyVar = RegOrdVar + OrdCatVar
        DepVar = FirstDepVar
        
        var_marginal_dataframe, base_to_Y_probability =getME(model_name,model_data,
                                                      dummyVar,ConVar,DepVar,
                                                      col_to_var,col_to_var_base,
                                                      test_size = test_size)
        # Var base --> Y1 base value name; Var specific --> Y1 non-base value name
        var_marginal_dataframe.columns=["Var base","Var specific",
                                'nocar','havecar']
        # Var base --> U- Y1 base value name; Var specific --> U- Y1 non-base value name        
        base_to_Y_probability.columns=["Var base","Var specific",
                                       "U-"+'nocar',"U-"+'havecar']
        direct_impacts_on_Y1=pd.concat([var_marginal_dataframe, base_to_Y_probability ],axis=1)

        direct_impacts_on_Y1.to_csv(model_name+"direct_impacts_on_Y1"+ ".csv")
        
        
        
        path_ana_value=["DirectMEonY2","uxForY1","MEonY1","uY1ForY2","MEofY1onY2","IndirectMEonY2","TotalMEonY2"]
        Path_Analysis_table=pd.DataFrame("",columns=["Var base","Var specific"]+path_ana_value,
                                 index=direct_impacts_on_Y2.index.tolist())
        
        
        Path_Analysis_table["Var base"]=direct_impacts_on_Y2["Var base"].iloc[:,0]
        Path_Analysis_table["Var specific"]=direct_impacts_on_Y2["Var specific"].iloc[:,0]
        # 'car' need to be changed to Y2's non-base value name
        Path_Analysis_table["DirectMEonY2"]=direct_impacts_on_Y2['car']  
        # the two 'nocar's would be changed to Y1's base value name(?)
        Path_Analysis_table["uxForY1"]=direct_impacts_on_Y1["U-"+'nocar']
        Path_Analysis_table["MEonY1"]=direct_impacts_on_Y1['nocar']
        # 'carcount@1' need to be changed to the Y1 name in direct_impacts_on_Y2 table
        # Two 'car's need to be changed to Y2'S non-base value name
        Path_Analysis_table["uY1ForY2"]=direct_impacts_on_Y2.loc['carcount@1',"U-"+'car']
        Path_Analysis_table["MEofY1onY2"]=direct_impacts_on_Y2.loc['carcount@1','car']        
        
        Path_Analysis_table["IndirectMEonY2"]= (Path_Analysis_table["uxForY1"].apply(pd.to_numeric)+Path_Analysis_table["MEonY1"].apply(pd.to_numeric))*\
                                                          (Path_Analysis_table["uY1ForY2"].apply(pd.to_numeric)+Path_Analysis_table["MEofY1onY2"].apply(pd.to_numeric))-\
                                                          Path_Analysis_table["uxForY1"].apply(pd.to_numeric)*Path_Analysis_table["uY1ForY2"].apply(pd.to_numeric) 
        
        
        Path_Analysis_table["TotalMEonY2"]=Path_Analysis_table["DirectMEonY2"] + Path_Analysis_table["IndirectMEonY2"]
        
        
        Path_Analysis_table.to_csv(model_name+".csv")
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        