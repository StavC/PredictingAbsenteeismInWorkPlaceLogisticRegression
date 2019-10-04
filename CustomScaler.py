from sklearn.base import  BaseEstimator,TransformerMixin
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

class CustomScaler(BaseEstimator,TransformerMixin):

    def __init__(self,columns,copy=True,with_mean=True,with_std=True):
        self.scaler=StandardScaler(copy,with_mean,with_std)
        self.columns=columns
        self.mean_=None
        self.var_=None

    def fit(self,X,y=None):
        self.scaler.fit(X[self.columns],y)
        self.mean_=np.mean(X[self.columns])
        self.var_=np.var(X[self.columns])
        return self

    def transform(self,X,y=None,copy=None):
        init_col_order=X.columns
        X_scaled=pd.DataFrame(self.scaler.transform(X[self.columns]),columns=self.columns)
        X_not_scaled=X.loc[:,~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled,X_scaled],axis=1)[init_col_order]

