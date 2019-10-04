import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LogisticRegression
from sklearn import metrics
from CustomScaler import CustomScaler
import pickle

def main():
    #### we want to predict Bsenteeism in workplace for empolyess
    raw_csv_data=pd.read_csv('Absenteeism-data.csv')
    df=raw_csv_data.copy()
    pd.options.display.max_rows=None
    pd.options.display.max_columns=None
    #print(df.head())
    #print(df.info())
    df=df.drop(['ID'],axis=1) #drop the ID COLUMN

    #### Handling the Reason for Absnece and Grouping
    reason_columns=pd.get_dummies(df['Reason for Absence'],drop_first=True)
    df=df.drop(['Reason for Absence'],axis=1)
    reason_type_1=reason_columns.loc[:,1:14].max(axis=1)
    reason_type_2=reason_columns.loc[:,15:17].max(axis=1)
    reason_type_3=reason_columns.loc[:,18:21].max(axis=1)
    reason_type_4=reason_columns.loc[:,22:].max(axis=1)

    ####concatenate the column values

    df=pd.concat([df,reason_type_1,reason_type_2,reason_type_3,reason_type_4],axis=1)
    #print(df)
    #print(df.columns.values)### change the name of 0 1 2 3
    column_names=['Date','Transportation Expense','Distance to Work' ,'Age',
 'Daily Work Load Average' ,'Body Mass Index' ,'Education', 'Children' ,'Pets',
 'Absenteeism Time in Hours','Reason_1','Reason_2','Reason_3','Reason_4']
    df.columns=column_names
    #print(df.head())

    ####changing the order
    column_names_reorderd=['Reason_1','Reason_2','Reason_3','Reason_4','Date','Transportation Expense','Distance to Work' ,'Age',
 'Daily Work Load Average' ,'Body Mass Index' ,'Education', 'Children' ,'Pets',
 'Absenteeism Time in Hours']
    df=df[column_names_reorderd]
    #print(df.head())

    df_reason_mod=df.copy() ##making a checkpoint

    #### handling the date column

    df_reason_mod['Date']=pd.to_datetime(df_reason_mod['Date'],format='%d/%m/%Y')
    #print(df_reason_mod['Date'])


    ####Extract the Month Value
    #### maybe the month has something to do with the empolyee absent

    list_months=[]
    for i in range(len(df_reason_mod['Date'])):
        list_months.append(df_reason_mod['Date'][i].month)

    #print(len(list_months))
    df_reason_mod['Month Value']=list_months
    #print(df_reason_mod.head(20))

    #### Extract the day of the week
    #### maybe friday can be a low day for ex

    def date_to_weekday(date):
        return date.weekday()

    df_reason_mod['Day of the Week']=df_reason_mod['Date'].apply(date_to_weekday)
    #print(df_reason_mod.head())

    #### combine Education to two groups one for high school and one for more than high school
    df_reason_mod['Education']=df_reason_mod['Education'].map({1:0,2:1,3:1,4:1})
    #print(df_reason_mod['Education'].value_counts())
    df_reason_mod=df_reason_mod.drop(['Date'],axis=1)
    df_reason_mod.to_csv('AbsProcced.csv')

    data=pd.read_csv('AbsProcced.csv')

    ################## MACHINE LEARNING START

    #### CREATE THE TARGETS
    median=data['Absenteeism Time in Hours'].median()
    #print(median)
    targets=np.where(data['Absenteeism Time in Hours']>median,1,0) ## if person absent is above the median
    #print(targets)
    data['Excessive Absenteeism']=targets
    #print(data.head())

    ### check if the targets are balanced
    #print(targets.sum()/targets.shape[0]) #0.45 is ok (45 to 55)

    data_with_targets=data.drop(['Absenteeism Time in Hours','Day of the Week','Daily Work Load Average','Distance to Work','Unnamed: 0'],axis=1)# not need this anymore
    print(data_with_targets.head())


    #### Select the inputs for the regression

    unscaled_inputs=data_with_targets.iloc[:,:-1] #getting all the inputs


    ##### Standardize the data


    columns_to_omit=['Reason_1','Reason_2','Reason_3','Reason_4','Education']
    columns_to_scale=[x for x in unscaled_inputs.columns.values if x not in columns_to_omit]
    absenteeism_scaler=CustomScaler(columns_to_scale)
    absenteeism_scaler.fit(unscaled_inputs)
    scaled_inputs=absenteeism_scaler.transform(unscaled_inputs)

    #print(scaled_inputs)

    #### Split the data into train and test

    x_train,x_test,y_train,y_test=train_test_split(scaled_inputs,targets,train_size=0.8,random_state=20)
    print(x_train.shape,y_train.shape)
    print(x_test.shape,y_train.shape)

    #### Traning the Model

    reg=LogisticRegression()
    reg.fit(x_train,y_train)
    print(reg.score(x_train,y_train))

    #### Manually check the accuracy
    model_outputs=reg.predict(x_train)
    model_targets=y_train
    #print(model_outputs==model_targets)
    print(np.sum(model_targets==model_outputs)/model_outputs.shape[0])

    #### Summary Table

    summary_table=pd.DataFrame(columns=['Feature Name'], data=unscaled_inputs.columns.values)
    summary_table['Coefficient']=np.transpose(reg.coef_)
    #print(summary_table)
    summary_table.index=summary_table.index+1
    summary_table.loc[0]=['Intercept',reg.intercept_[0]]
    summary_table=summary_table.sort_index()
    #print(summary_table)# coefficient is also a weight and intercpet is also a bias
    summary_table['Odds_ratio']=np.exp(summary_table.Coefficient)
    #print(summary_table)
    summary_table=summary_table.sort_values('Odds_ratio',ascending=False)
    print(summary_table) ### most important at the top

    #### TESTING THE MACHINE LEARNING WITH DATA

    print(reg.score(x_test,y_test))
    predicted_proba=reg.predict_proba(x_test)
    print(predicted_proba)# left is probality of getting 0 and right probality of getting 1 from the model


    #### SAVE THE MODEL

    with open('model','wb') as file:
        pickle.dump(reg,file)
    with open('Scaler','wb') as file:
        pickle.dump(absenteeism_scaler,file)












if __name__ == '__main__':
    main()