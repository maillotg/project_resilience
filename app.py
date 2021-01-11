import streamlit as st

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import cross_val_score

from xgboost import XGBRegressor


class CustomScaler(TransformerMixin, BaseEstimator): 
# TransformerMixin generates a fit_transform method from fit and transform
# BaseEstimator generates get_params and set_params methods
    def __init__(self):
        pass
        
    
    
    def fit(self, X, y=None):
        self.means = X.mean()
        self.max = X.max()
        self.min = X.min()
        return self
    
    def transform(self, X, y=None):
        X= 100 - X
        X_transformed = (X - self.min)/(self.max-self.min) 
        # Return result as dataframe for integration into ColumnTransformer
        return pd.DataFrame(X_transformed)

st.markdown("""# Resilience Project
## This is a sub header
This is text""")

data_path="raw_data/CDP-Cities-KPI.csv"
flo= ['Nb.Hazards.Type', 'Hazards.Exposure.Level',
       'Adaptation.Challenges.Health', 'Adaptation.Challenges.Economic',
       'Adaptation.Challenges.Environment',
       'Adaptation.Challenges.Infrastructure', 'Adaptation.Challenges.Social',
       'Adaptation.Challenges.Governance', 'Adaptation.Challenges.Education',
       'Adaptation.Challenges.Level', 'Electricity.Source.Biomass',
       'Electricity.Source.Coal', 'Electricity.Source.Gas',
       'Electricity.Source.Geothermal', 'Electricity.Source.Hydro',
       'Electricity.Source.Nuclear', 'Electricity.Source.Oil',
       'Electricity.Source.Other', 'Electricity.Source.Solar',
       'Electricity.Source.Wind', 'Electricity.Source.Renewable',
       'Transport.Mode.Passenger.Public', 'Transport.Mode.Passenger.Cycling',
       'Transport.Mode.Passenger.Other',
       'Transport.Mode.Passenger.Private.motorized',
       'Transport.Mode.Passenger.Walking']
cat=['Sustainability.Targets.Master.Planning',
       'Risk.Assessment.Actions', 'Risk.Health.System', 'Adaptation.Plan',
       'City.Wide.Emissions.Inventory', 'GHG.Emissions.Primary.protocol',
       'GHG.Emissions.Evolution', 'GHG.Emissions.Consumption',
       'GHG.Emissions.External.Verification',
       'GHG.Emissions.Reductions.Targets',
       'Emissions.Reductions.Mitigation.Planning',
       'Opportunities.Collaboration', 'Renewable.Energy.Target',
       'Energy.Efficnecy.Target', 'Low.Zero.Emission.Zone',
       'Food.Consumption.Policies', 'Water.Resource.Management.strategy']
cri = pd.read_csv('raw_data/CRI.csv')
vul = pd.read_csv('notebooks/vulnerability.csv')
readiness=cri['City.Readiness.Index']
v=vul['Vulnerability']
#st.write(v)

col_to_use=['Hazards.Exposure.Level',
'Adaptation.Plan',
'Adaptation.Challenges.Level',
'GHG.Emissions.Reductions.Targets',
'GHG.Emissions.Consumption',
'Emissions.Reductions.Mitigation.Planning',
'Potable.Water.Supply.Percent',
'Sustainability.Targets.Master.Planning',
'Risk.Assessment.Actions',
'Risk.Health.System',
'Low.Zero.Emission.Zone']
col_to_use=[
'Hazards.Exposure.Level',
'Risk.Health.System',
'Adaptation.Plan',
'Emissions.Reductions.Mitigation.Planning',
'Water.Resource.Management.strategy',
'GHG.Emissions.Reductions.Targets',
'Food.Consumption.Policies',
'Potable.Water.Supply.Percent',
'Adaptation.Challenges.Level',
'Electricity.Source.Renewable']

@st.cache
def load_data():
	data = pd.read_csv(data_path)
	return data





data=load_data()
#X=data[flo+cat]
#X=data.drop(['Organization','City','Country','Unnamed: 0', 'Year.Reported.to.CDP', 'Account.Number','CDP.Region', 'First.Time.Discloser','Country.Code.3','City.Location'],1)
X=data[col_to_use]
st.write(X)

y = (readiness*v)/(readiness+v)
#y = readiness
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state=1)


reg=LinearRegression()
reg = XGBRegressor()
customscaler = CustomScaler()

water_transform = Pipeline([('imputer', SimpleImputer(strategy = 'constant',fill_value=0.0)),
                            ('scaler', customscaler)])

num_transformer = Pipeline([('imputer', SimpleImputer(strategy = 'constant',fill_value=0)),
                            ('scaler', MinMaxScaler())])

cat_transformer = Pipeline( [ ('imputer',SimpleImputer(strategy='constant',fill_value='No')),
                            ('scaler',OneHotEncoder(handle_unknown='ignore'))
                            ])

preprocessor = ColumnTransformer([
    ('water_transformer', water_transform,['Potable.Water.Supply.Percent']),
    ('num_transformer', num_transformer,  make_column_selector(dtype_include=['float64'])),
    ('cat_transformer', cat_transformer,  make_column_selector(dtype_include=['object']))
    
    ])


final_pipel = Pipeline([
    ('preprocessing', preprocessor),
    ('linear_regression', reg)])




final_pipe_trained = final_pipel.fit(X_train,y_train)

# Make predictions
#final_pipe_trained.predict(X_test.iloc[0:2])

# Score model
st.write(final_pipe_trained.score(X_test,y_test))
st.write(cross_val_score(final_pipel, X_train, y_train, cv=8, scoring='r2').mean())



option = st.selectbox(
     'How would you like to be contacted?',
     ('Yes', 'No', 'Do not know'),key="4")

st.write('You selected:', option)

option = st.selectbox(
     'How would you like to be contacted?',
     ('Yes', 'No', 'Do not know'),key="5")

st.write('You selected:', option)

option = st.selectbox(
     'How would you like to be contacted?',
     ('Yes', 'No', 'Do not know'),key="6")

st.write('You selected:', option)

option = st.selectbox(
     'How would you like to be contacted?',
     ('Yes', 'No', 'Do not know'),key="1")

st.write('You selected:', option)

option = st.selectbox(
     'How would you like to be contacted?',
     ('Yes', 'No', 'Do not know'),key="2")

st.write('You selected:', option)

option = st.selectbox(
     'How would you like to be contacted?',
     ('Yes', 'No', 'Do not know'),key="3")

st.write('You selected:', option)




















