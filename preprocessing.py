import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import scipy.stats as stat
import pylab

def remove_outliers(df,feature,upper_limit,lower_limit):  
  q1=df[feature].mean()-0.675*df[feature].std()
  q3=df[feature].mean()+0.675*df[feature].std()
  iqr=q3-q1
  max=q3+upper_limit*iqr
  min=q1-lower_limit*iqr
  df[feature]=np.where(df[feature]>max,max,df[feature])

def scale_test_point(test_point):

    temp=pd.read_csv('heart.csv')
    original_columns=temp.columns
    del temp

    original_data=pd.read_csv('heart.csv')
    test_point.append(1)
    test_data=np.array(test_point).reshape((1,-1))
    test_data=pd.DataFrame(test_data,columns=original_columns)

    df=pd.concat([test_data,original_data])


    df['trestbps']=df['trestbps'].astype(float)
    df['trestbps'],parameters=stat.boxcox(df['trestbps'])
    remove_outliers(df,'trestbps',1.5,1.5)

    df['chol']=df['chol'].astype(float)
    df['chol'],parameters=stat.boxcox(df['chol'])
    remove_outliers(df,'chol',1.5,1.5)

    df['thalach']=df['thalach'].astype(float)
    df['thalach'],param=stat.boxcox(df['thalach'])

    ca_mapper=dict({0:175,1:65,2:38,3:20,4:5})
    df['ca']=df.ca.map(ca_mapper)

    thal_mapper=dict({2:166,3:177,1:18,0:2})
    df['thal']=df.thal.map(thal_mapper)

    a = pd.get_dummies(df['cp'], prefix = "cp")
    c = pd.get_dummies(df['slope'], prefix = "slope")
    frames = [df, a,c]
    df = pd.concat(frames, axis = 1)
    df = df.drop(columns = ['cp', 'slope'])

    X=df.drop('target',axis=1)
    y=df['target']
    X=pd.DataFrame(StandardScaler().fit_transform(X),columns=X.columns)

    test_point=np.array(X.iloc[0])

    trained_model = joblib.load('./LogisticRegression.pkl')

    test_point=pd.DataFrame(test_point.reshape((1,-1)))
    print(test_point)
    prediction = trained_model.predict(test_point.fillna(0))
    print(prediction)
    return prediction