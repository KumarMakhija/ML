# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 12:05:37 2018

@author: arpit.agarwal
"""
import pandas as pd
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns

Original_train = pd.read_csv("E:/Unilever Singapore (15-AIM-1275)/Arpit/PUBG/train_V2.csv")
Original_test = pd.read_csv("E:/Unilever Singapore (15-AIM-1275)/Arpit/PUBG/test_V2.csv")

Original_train = Original_train.dropna() #droping rows with NA values
Original_test = Original_test.dropna() #droping rows with NA values

f = lambda x: "Solo" if "solo" in x else ("Duo" if ("duo" in x) or ("crash" in x) else "Squad")
Original_train['Team'] = Original_train.matchType.apply(f)
Original_test['Team'] = Original_test.matchType.apply(f)

## =============================================================================
#####################################################################################
#####################################################################################

def Feature_Engineering(Data_set):

    Data_set['Avg_Ranking'] = (Data_set['winPoints'] + Data_set['killPoints'])/2
    Data_set['healthItems'] = Data_set.heals + Data_set.boosts
    Data_set['total_items'] = Data_set.boosts + Data_set.heals + Data_set.weaponsAcquired
    
    Data_set['total_distance'] = Data_set['walkDistance'] + Data_set['swimDistance'] + Data_set['rideDistance']
             
    Data_set["specialKills"] = Data_set.headshotKills + Data_set.roadKills
    
    features = Data_set.drop(columns = ['Id', 'groupId', 'matchId','Team', 
                                         'matchType']).columns.tolist()
    
    Data_set['playersJoined'] = Data_set.groupby('matchId')['matchId'].transform('count')
    
    Data_set['team_size'] = Data_set.groupby(['matchId','groupId']).groupId.transform('count')
    Data_set['match_size']=Data_set.groupby('matchId').Id.transform('nunique')
   
    Data_set['healthitems_per_WD'] = Data_set.healthItems/(Data_set.walkDistance +1)
    Data_set['healthitems_per_TD'] =  Data_set.healthItems/(Data_set.total_distance+1)
    
    Data_set['boosts_per_WD'] =  Data_set.boosts /(Data_set.walkDistance +1) 
    Data_set['boosts_per_TD'] =  Data_set.boosts /(Data_set.total_distance+1)
    
    Data_set['kills_per_WD'] = Data_set.kills / (Data_set.walkDistance+1) 
    Data_set['kills_per_TD'] = Data_set.kills / (Data_set.total_distance+1)
    
    Data_set['weapons_per_WD'] =  Data_set.weaponsAcquired/(Data_set.walkDistance+1) 
    Data_set['weapons_per_TD'] =  Data_set.weaponsAcquired/(Data_set.total_distance+1)
    
    Data_set['specialKills_per_WD'] = Data_set.specialKills/(Data_set.walkDistance+1) 
    Data_set['specialKills_per_TD'] = Data_set.specialKills/(Data_set.total_distance+1)
    
    Data_set['knocked_per_TD'] = Data_set.DBNOs/(Data_set.total_distance+1)
    Data_set['damage_per_TD'] = Data_set.damageDealt/(Data_set.total_distance+1)
    Data_set['revives_per_TD'] = Data_set.revives/(Data_set.total_distance+1)
    
    Data_set['killsWithoutMoving'] = ((Data_set.kills > 0) & (Data_set.total_distance == 0))
    
    bool_change = lambda x: int(x)
    Data_set['killsWithoutMoving'] = Data_set.killsWithoutMoving.apply(bool_change)
    
    
    Data_set['killPlaceOverMaxPlace'] = Data_set.killPlace/(Data_set.maxPlace+2)
    
    Data_set['headshot_kill_rate'] = Data_set.headshotKills/(Data_set.kills+1)
    
    Data_set['max_possible_kills'] = Data_set.match_size - Data_set.team_size
    
    Data_set['pct_killed'] = Data_set.kills/(Data_set.max_possible_kills+1)
    Data_set['pct_knocked'] = Data_set.DBNOs/(Data_set.max_possible_kills+1)
    
    Data_set['map_has_sea'] =  Data_set.groupby('matchId').swimDistance.transform('sum').apply(lambda x: 1 if x>0 else 0)
       
    def naming(Data_set1, j):
        list1 = ['matchId','groupId']
        num_col = len(Data_set1.columns)
        list1.extend(Data_set1.iloc[:,2:num_col].columns +"_"+ j)
        Data_set1.columns = list1
        return Data_set1
    
    group_mean = naming(Data_set.groupby(['matchId','groupId'],as_index=False)[features].agg('mean'),"group_mean")
    group_max = naming(Data_set.groupby(['matchId','groupId'],as_index=False)[features].agg('max'),"group_max")
    group_min = naming(Data_set.groupby(['matchId','groupId'],as_index=False)[features].agg('min'),"group_minn")
    
    match_mean = naming(Data_set.groupby(['matchId'],as_index=False)[features].agg('mean'),"match_mean")
    match_max = naming(Data_set.groupby(['matchId'],as_index=False)[features].agg('max'),"match_max")
    match_min = naming(Data_set.groupby(['matchId'],as_index=False)[features].agg('min'),"match_min")

    Data_set = pd.merge(Data_set,group_mean, how= 'left',on= ['matchId','groupId'])
    Data_set = pd.merge(Data_set,group_max, how= 'left',on= ['matchId','groupId'])
    Data_set = pd.merge(Data_set,group_min, how= 'left',on= ['matchId','groupId'])
    Data_set = pd.merge(Data_set,match_mean, how= 'left',on= ['matchId'])
    Data_set = pd.merge(Data_set,match_max, how= 'left',on= ['matchId'])
    Data_set = pd.merge(Data_set,match_min, how= 'left',on= ['matchId'])
    
    Data_set.fillna(0, inplace=True)
    return Data_set

## ============================================================================
###############################################################################
###############################################################################

df_train = Feature_Engineering(Original_train.drop(columns = ['winPlacePerc'])).drop(columns = ['matchType'])
df_train = pd.concat([df_train,Original_train['winPlacePerc']], axis = 1)

#test = Original_train[(Original_train.groupId == '0e009e443d9c0d') | (Original_train.groupId == '0b4bf93ca082a4')] 

#test1  = Feature_Engineering(test.drop(columns = ['winPlacePerc']))


df_test = Feature_Engineering(Original_test).drop(columns = ['matchType'])

del Original_train
del Original_test

## =============================================================================
#####################################################################################
#####################################################################################


#df_train[["Team","DBNOs"]].groupby("Team").sum().plot(kind = "bar")
#df_train[["Team","revives"]].groupby("Team").sum().plot(kind = "bar")
#df_train[["Team","total_distance_team"]].groupby("Team").sum().plot(kind = "bar")
#df_train[["Team","knocked_per_distance"]].groupby("Team").sum().plot(kind = "bar")
#df_train[["Team","revives_per_distance"]].groupby("Team").sum().plot(kind = "bar")
#df_train[["Team","pct_knocked"]].groupby("Team").sum().plot(kind = "bar") 



## =============================================================================
#
#def correlation_table(dataset):
#    return round(dataset.corr(),2)
#
#cor_Solo = correlation_table(Solo_train.drop(columns = ['winPlacePerc']))
#cor_Duo = correlation_table(Duo_train.drop(columns = ['winPlacePerc']))
#cor_Squad = correlation_table(Squad_train.drop(columns = ['winPlacePerc']))
#
#
#
#cor_Solo.to_csv('E:/Unilever Singapore (15-AIM-1275)/Arpit/PUBG/cor_Solo.csv')
#cor_Duo.to_csv('E:/Unilever Singapore (15-AIM-1275)/Arpit/PUBG/cor_Duo.csv')
#cor_Squad.to_csv('E:/Unilever Singapore (15-AIM-1275)/Arpit/PUBG/cor_Squad.csv')
#

#Solo_train.to_csv('E:/Unilever Singapore (15-AIM-1275)/Arpit/PUBG/Solo_train.csv')
#Solo_test.to_csv('E:/Unilever Singapore (15-AIM-1275)/Arpit/PUBG/Solo_test.csv')


#====================================================================================
#####################################################################################
#####################################################################################

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


def factor_analysis(train_set, test_set):
    train_df = pd.DataFrame(scale(train_set.drop(columns = ['winPlacePerc','Id', 'groupId', 'matchId'])))
    test_df = pd.DataFrame(scale(test_set.drop(columns = ['Id', 'groupId', 'matchId'])))
    pca = PCA(n_components = 61).fit(train_df)
    
    #Cumulative Variance explains
    var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
    no_comp = sum(var1 <95) +1
    pca = PCA(n_components=no_comp)
    plt.plot(var1)

    train_df = pca.fit_transform(train_df)
    test_df = pca.fit_transform(test_df)
    
    list1 = [str(i) for i in range(1,no_comp+1)]
    list1 = ["Comp"+ x for x in list1]
    train_df = pd.DataFrame(train_df, columns = list1)
    test_df = pd.DataFrame(test_df, columns = list1)
    train_df  = pd.concat([train_set.iloc[:,0:3].reset_index(drop=True),
                           train_df,
                           train_set['winPlacePerc'].reset_index(drop=True)], axis = 1)
    test_df  = pd.concat([test_set.iloc[:,0:3].reset_index(drop=True),
                          test_df], axis = 1)
    del list1
    del pca
    del no_comp
    del var1
    return train_df, test_df

#######==============================================================================
#####################################################################################
#####################################################################################
def uncorrelated_features(train_data, test_data,Team):    
    if Team == "Solo" :
        features = ['_specialKills','assists',
                    'Avg Ranking','damage_per_distance',
                    'damageDealt','headshot_kill_rate',
                    'healthitems_per_total_distance','kills_per_total_distance',
                    'killStreaks','killsWithoutMoving',
                    'longestKill','map_has_sea',
                    'matchDuration','vehicleDestroys',
                    'max_possible_kills','max_team_size',
                    'pct_killed','roadKills',
                    'specialKills_per distance',
                    'swimDistance','teamKills',
                    'total_distance','total_items_acquired',
                    'weaponsAcquired_per__total_distance']
        train_data = pd.concat([train_data[features].reset_index(drop=True),
                                train_data['winPlacePerc'].reset_index(drop=True)], axis =1 )
        test_data = test_data[features].reset_index(drop=True)
    elif Team == "Duo":
         features = ['_specialKills','assists',
                    'Avg Ranking','damage_per_distance',
                    'damageDealt','headshot_kill_rate',
                    'healthitems_per_total_distance','kills_per_total_distance',
                    'killStreaks','killsWithoutMoving',
                    'longestKill','map_has_sea',
                    'matchDuration','vehicleDestroys',
                    'max_possible_kills','max_team_size',
                    'pct_killed','roadKills',
                    'specialKills_per distance',
                    'swimDistance','teamKills',
                    'total_distance','total_items_acquired',
                    'weaponsAcquired_per__total_distance',
                    'boosts_per_walkDistance','DBNOs',
                    'kills','knocked_per_distance',
                    'pct_knocked','pct_team_killed',
                    'revives','revives_per_distance',
                    'team_kill_points','team_kill_rank',
                    'team_size','total_distance_team',
                    'total_team_damage','total_team_weaponsAcquired']
         train_data = pd.concat([train_data[features].reset_index(drop=True),
                                train_data['winPlacePerc'].reset_index(drop=True)], axis =1 )
         test_data = test_data[features].reset_index(drop=True)    
    else:
         features = ['_specialKills','assists',
                    'Avg Ranking','damage_per_distance',
                    'damageDealt','headshot_kill_rate',
                    'healthitems_per_total_distance','kills_per_total_distance',
                    'killStreaks','killsWithoutMoving',
                    'longestKill','map_has_sea',
                    'matchDuration','vehicleDestroys',
                    'max_possible_kills','max_team_size',
                    'pct_killed','roadKills',
                    'specialKills_per distance',
                    'swimDistance','teamKills',
                    'total_distance','total_items_acquired',
                    'weaponsAcquired_per__total_distance',
                    'boosts_per_walkDistance','DBNOs',
                    'kills','knocked_per_distance',
                    'pct_knocked','pct_team_killed',
                    'revives','revives_per_distance',
                    'team_kill_points','team_kill_rank',
                    'team_size','total_distance_team',
                    'total_team_damage','total_team_weaponsAcquired',
                    '_killPlaceOverMaxPlace','max_kills_by_team',
                    'rankPoints','specialKills_per distance',
                    'total_team_healthItems','total_team_kills']
         train_data = pd.concat([train_data[features].reset_index(drop=True),
                                train_data['winPlacePerc'].reset_index(drop=True)], axis =1 )
         test_data = test_data[features].reset_index(drop=True)  
    
    return train_data,test_data


#####################################################################################
#####################################################################################

def useful_features(train_data, test_data):
    correlation = pd.DataFrame(pd.to_numeric(train_data.corr()['winPlacePerc']))
    correlation['winPlacePerc'] = pd.to_numeric(train_data.corr()['winPlacePerc'])
    
    features  = correlation[(pd.to_numeric(correlation['winPlacePerc']) > 0.1) |
                            (pd.to_numeric(correlation['winPlacePerc']) < -0.1)].index.drop('winPlacePerc')
    
    train_data_unused = train_data.iloc[:,0:3].reset_index(drop=True)
    train_data_target = train_data['winPlacePerc'].reset_index(drop=True)
    train_data1 = train_data[features].reset_index(drop=True)
    train_data1 = pd.concat([train_data_unused,train_data1,train_data_target], axis = 1)
    
    test_data_unused = test_data.iloc[:,0:3].reset_index(drop=True)
    test_data1 = test_data[features].reset_index(drop=True)
    test_data1 = pd.concat([test_data_unused,test_data1], axis = 1)
    
    return train_data1,test_data1

## =============================================================================
#####################################################################################
#####################################################################################

def target_split(train_Data_set, test_Data_set):
    train_X = train_Data_set.drop(['Id','groupId_x','matchId','winPlacePerc'], axis = 1)
    train_Y = train_Data_set['winPlacePerc']
    
    test_X = test_Data_set.drop(['Id','groupId_x','matchId'], axis = 1)
    Submission = test_Data_set['Id']
    return train_X,train_Y,test_X, Submission

## =============================================================================
#####################################################################################
#####################################################################################

Solo_train = df_train[df_train['Team'] == 'Solo'].drop(columns = ['Team'])
Solo_test = df_test[df_test['Team'] == 'Solo'].drop(columns = ['Team'])
#factored_Solo_train, factored_Solo_test= factor_analysis(Solo_train, Solo_test)
#Solo_train, Solo_test = uncorrelated_features(Solo_train, Solo_test, "Solo")
Solo_train, Solo_test = useful_features(Solo_train, Solo_test)




Duo_train = df_train[df_train['Team'] == 'Duo'].drop(columns = ['Team'])
Duo_test = df_test[df_test['Team'] == 'Duo'].drop(columns = ['Team'])
#factored_Duo_train , factored_Duo_test= factor_analysis(Duo_train , Duo_test)
#Duo_train, Duo_test = uncorrelated_features(Duo_train, Duo_test, "Duo")
#Duo_train, Duo_test = useful_features(Duo_train, Duo_test)




Squad_train = df_train[df_train['Team'] == 'Squad'].drop(columns = ['Team']) 
Squad_test = df_test[df_test['Team'] == 'Squad'].drop(columns = ['Team']) 
#factored_Squad_train, factored_Squad_test= factor_analysis(Squad_train, Squad_test)
#Squad_train, Squad_test = useful_features(Squad_train, Squad_test)
#Squad_train, Squad_test = uncorrelated_features(Squad_train, Squad_test, "Squad")

def scatter(x):
    df_train.plot(x=x,y="winPlacePerc", kind="scatter", figsize = (8,6))
    
scatter("teamKills_group_mean")

#####################################################################################
#####################################################################################
#####################################################################################
import xgboost as xgb
import lightgbm as lgb
import gc
gc.enable()
#from sklearn.metrics import mean_absolute_error
        

matchType = ['Solo','Duo', 'Sqaud']

for i in matchType:
    if i == "Solo" :
        train_X, train_Y, test_X, Submission_Solo = target_split(Solo_train, Solo_test)

        train_index = round(int(train_X.shape[0]*0.8))
        dev_X = train_X[:train_index] 
        val_X = train_X[train_index:]
        dev_y = train_Y[:train_index] 
        val_y = train_Y[train_index:] 
        gc.collect();
            
            # custom function to run light gbm model
        def run_lgb(train_X, train_y, val_X, val_y, x_test):
            params = {"objective" : "regression", "metric" : "mae",
                      'n_estimators':20000,"learning_rate" : 0.1, 
                      "bagging_fraction" : 0.6, "bagging_seed" : 0,
                      "min_data_in_leaf": 5, "max_depth" : 15,
                      "num_threads" : 15,"colsample_bytree" : 0.6}           
                
            lgtrain = lgb.Dataset(train_X, label=train_y)
            lgval = lgb.Dataset(val_X, label=val_y)
            model = lgb.train(params, 
                              lgtrain, 
                              valid_sets=[lgtrain, lgval], 
                              early_stopping_rounds=1000, 
                              verbose_eval=1000)
                        
            pred_test_y = model.predict(x_test, num_iteration=model.best_iteration)
            return pred_test_y, model
            
            # Training the model #
            
        Solo_test_Y, model = run_lgb(dev_X, dev_y, val_X, val_y, test_X)

        
        feature_importance = pd.DataFrame(100.0*(model.feature_importance()/
                                                 model.feature_importance().sum()),
            index = model.feature_name(),
            columns = ["Importance"]).sort_values("Importance", ascending = False)
        
        feature_importance = pd.DataFrame(pd.Series(feature_importance.Importance).cumsum())
        
        features = pd.Series(feature_importance[feature_importance['Importance']<95].index)
        dev_X = dev_X[features]
        val_X = val_X[features]
        test_X = test_X[features]
        Solo_test_Y, model_1 = run_lgb(dev_X, dev_y, val_X, val_y, test_X)

        Submission_Solo = pd.concat([Submission_Solo.reset_index(drop=True),pd.DataFrame(Solo_test_Y,columns=['winPlacePerc'])], axis = 1)
        
    elif i == "Duo":
        train_X, train_Y, test_X, Submission_Duo = target_split(Duo_train, Duo_test)

        train_index = round(int(train_X.shape[0]*0.8))
        dev_X = train_X[:train_index] 
        val_X = train_X[train_index:]
        dev_y = train_Y[:train_index] 
        val_y = train_Y[train_index:] 
        gc.collect();
            
            # custom function to run light gbm model
        def run_lgb(train_X, train_y, val_X, val_y, x_test):
            params = {"objective" : "regression", "metric" : "mae",
                      'n_estimators':20000,"learning_rate" : 0.1, 
                      "bagging_fraction" : 0.7, "bagging_seed" : 0,
                      "min_data_in_leaf": 10, "max_depth" : 20,
                      "num_threads" : 15,"colsample_bytree" : 0.7}           
                
            lgtrain = lgb.Dataset(train_X, label=train_y)
            lgval = lgb.Dataset(val_X, label=val_y)
            model = lgb.train(params, 
                              lgtrain, 
                              valid_sets=[lgtrain, lgval], 
                              early_stopping_rounds=1000, 
                              verbose_eval=1000)
                        
            pred_test_y = model.predict(x_test, num_iteration=model.best_iteration)
            return pred_test_y, model        
        
        Duo_test_Y,model = run_lgb(dev_X, dev_y, val_X, val_y, test_X)

#        feature_importance = pd.DataFrame(100.0*(model.feature_importance()/
#                                                 model.feature_importance().sum()),
#            index = model.feature_name(),
#            columns = ["Importance"]).sort_values("Importance", ascending = False)
#        
#        feature_importance = pd.DataFrame(pd.Series(feature_importance.Importance).cumsum())
#        
#        features = pd.Series(feature_importance[feature_importance['Importance']<90].index)
#        dev_X = dev_X[features]
#        val_X = val_X[features]
#        test_X = test_X[features]
#        Duo_test_Y,model_1 = run_lgb(dev_X, dev_y, val_X, val_y, test_X)

        Submission_Duo = pd.concat([Submission_Duo.reset_index(drop=True),pd.DataFrame(Duo_test_Y,columns=['winPlacePerc'])], axis = 1)
        
    else:
        train_X, train_Y, test_X, Submission_Squad = target_split(Squad_train, Squad_test)
        train_index = round(int(train_X.shape[0]*0.8))
        dev_X = train_X[:train_index] 
        val_X = train_X[train_index:]
        dev_y = train_Y[:train_index] 
        val_y = train_Y[train_index:] 
        gc.collect();
            
            # custom function to run light gbm model
        def run_lgb(train_X, train_y, val_X, val_y, x_test):
            params = {"objective" : "regression", "metric" : "mae",
                      'n_estimators':20000,"learning_rate" : 0.3, 
                      "bagging_fraction" : 0.7, "bagging_seed" : 0,
                      "min_data_in_leaf": 30, "max_depth" : 15,
                      "num_threads" : 15,"colsample_bytree" : 0.7}           
                
            lgtrain = lgb.Dataset(train_X, label=train_y)
            lgval = lgb.Dataset(val_X, label=val_y)
            model = lgb.train(params, 
                              lgtrain, 
                              valid_sets=[lgtrain, lgval], 
                              early_stopping_rounds=500, 
                              verbose_eval=1000)
                        
            pred_test_y = model.predict(x_test, num_iteration=model.best_iteration)
            return pred_test_y, model                
        
        
        Squad_test_Y,model = run_lgb(dev_X, dev_y, val_X, val_y, test_X)
#
#        feature_importance = pd.DataFrame(100.0*(model.feature_importance()/
#                                                 model.feature_importance().sum()),
#            index = model.feature_name(),
#            columns = ["Importance"]).sort_values("Importance", ascending = False)
#        
#        feature_importance = pd.DataFrame(pd.Series(feature_importance.Importance).cumsum())
#        
#        features = pd.Series(feature_importance[feature_importance['Importance']<90].index)
#        dev_X = dev_X[features]
#        val_X = val_X[features]
#        test_X = test_X[features]
#        
#        Squad_test_Y,model_1 = run_lgb(dev_X, dev_y, val_X, val_y, test_X)
        
        Submission_Squad = pd.concat([Submission_Squad.reset_index(drop=True),pd.DataFrame(Squad_test_Y,columns=['winPlacePerc'])], axis = 1)
        
Submission_file = pd.concat([Submission_Solo,Submission_Duo,Submission_Squad], axis = 0)


Submission_file.to_csv('E:/Unilever Singapore (15-AIM-1275)/Arpit/PUBG/Submission_file_20190104 1700hrs.csv', index = False)
