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
from sklearn.neighbors import NearestNeighbors
import folium
from streamlit_folium import folium_static
from sklearn.linear_model import Ridge
st.markdown(
    f'''
        <style>
            .sidebar .sidebar-content {{
                width: 50%;
            }}
        </style>
    ''',
    unsafe_allow_html=True
)
def ssminmaxe(x):
    m=0
    M=352
    return (x-m)/(M-m)
def ssminmax(x):
    m=0
    M=8
    return (x-m)/(M-m)
def ssminma(x):
    m=0
    M=100
    return (x-m)/(M-m)
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
### Additional information:
Factors impacting your adaptation to Climate Change""")
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
#data= data.drop_duplicates(subset = 'Account.Number',keep = 'first',inplace =False).copy().reset_index()
#data = data.drop('index',axis=1)
new1= data.copy()
new1['vul']=v
new1['readiness']=readiness
data= new1.drop_duplicates(subset = 'Account.Number',keep = 'first',inplace =False).copy().reset_index()
data = data.drop('index',axis=1)
v=data['vul']
readiness=data['readiness']
#new1['Hazards.Exposure.Level']=new1['Hazards.Exposure.Level'].apply(cleanan)
#new1['Adaptation.Challenges.Level']=new1['Adaptation.Challenges.Level'].apply(cleanan)
#new1['Risk.Health.System']=new1['Risk.Health.System'].apply(strinan)
#new1['Potable.Water.Supply.vulnerability']=new1['Potable.Water.Supply.Percent'].apply(pournan)
#new1['exposure.level']=new1['Hazards.Exposure.Level'].apply(ssminmaxe)
#new1['City.Adaptation.Challenges.Index'] =new1['Adaptation.Challenges.Level'].apply(ssminmax)
#new1['Potable.Water.Supply.vulnerability']= new1['Potable.Water.Supply.vulnerability'].apply(ssminma)
#new1['sensitivity.index']=(new1['Risk.Health.System'] + new1['Potable.Water.Supply.vulnerability'])/2
#new1['Vulnerability']=(new1['exposure.level']+new1['City.Adaptation.Challenges.Index']+new1['sensitivity.index'])/3
#v= new1['Vulnerability']
#X=data[flo+cat]
#X=data.drop(['Organization','City','Country','Unnamed: 0', 'Year.Reported.to.CDP', 'Account.Number','CDP.Region', 'First.Time.Discloser','Country.Code.3','City.Location'],1)
X=data[col_to_use]
df = data.copy()
df['readiness']=readiness
df['vulnerability']=v
#st.write(X)
#X
k = (readiness*(1-v))/(readiness+(1-v))
#Our first y is vulnerability
y = v
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state=1)
#Lets get our other y by taking the same dataset from our splitting and taking then the index
#For readiness
Xr_train = X_train.copy()
Xr_test = X_test.copy()
yr_train = df['readiness'].iloc[Xr_train.index]
yr_test = df['readiness'].iloc[Xr_test.index]
#For resilience score which is our k in this script
Xk_train = X_train.copy()
Xk_test = X_test.copy()
yk_train = k.iloc[Xk_train.index]
yk_test = k.iloc[Xk_test.index]
reg=LinearRegression()
reg = XGBRegressor()
#reg = Ridge()
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
facto={"Economic":"Access to basic service, Cost of living, Poverty, Unemployment, Economic health, economic diversity, and Budgetary capacity.","Health":"Access to healthcare and Public health","Education":"Access to education","Habitat":"Housing","Infrastructure":"Rapid urbanization, Infrastructure conditions / maintenance, and Infrastructure capacity","Social":"Inequality and Migration","Environment":"Resource availability, Environmental conditions","Governance":"safety and security,political engagement ,transparency"}
# Score model
st.write("List of sectors compiling the different factors")
st.write(facto)
#st.write(final_pipe_trained.score(X_test,y_test))
#st.write(cross_val_score(final_pipel, X_train, y_train, cv=8, scoring='r2').mean())
st.sidebar.write('Survey')
challenlevel = st.sidebar.slider("How many factors (list on your right) impact your city on adapting to Climate Change ? If one or several factors are in the same sector just count them as 1.", 0, 8, 1)
q_water =  st.sidebar.slider("What is the percentage of your city’s population having access to potable water supply service ?",0,100,1)
elec_source_renew= st.sidebar.slider("How much from your energy mix is coming from renewable energy ?",0,100,1)
hazardexpolvl = st.sidebar.slider("How vulnerable is your city to Climate Hazards ? List of climate hazards : Extreme Precipitation, Extreme Storm and Wind, Extreme Temperature, Flood and Sea level rise, Biological Hazards, Wild fires, Water Scarcity, Mass Movement (avalanche etc.)If you have had one of this hazards, count 50 and adds them up)",0,350,0,50)
option1 = st.sidebar.selectbox(
     'Does your city have a risk health system?',
     ('Yes', 'No', 'Do not know'),key="4")
#st.write('You selected:', option1)
option2 = st.sidebar.selectbox(
     "Does your city have an adaptation plan?",
     ('Yes', 'No', 'Do not know'),key="5")
option3 = st.sidebar.selectbox(
     "Does your city have a climate change mitigation or energy access plan for reducing city-wide GHG emissions?",
     ('Yes', 'No', 'Do not know'),key="6")
option4 = st.sidebar.selectbox(
     "Does your city have a publicly available Water Resource Management strategy?",
     ('Yes', 'No', 'Do not know'),key="1")
option5 = st.sidebar.selectbox(
     "Do you have a GHG emissions reduction target(s) in place at the city-wide level?",
     ('Yes', 'No', 'Do not know'),key="2")
option6 = st.sidebar.selectbox(
     "Does your city have any policies relating to food consumption within your city?",
     ('Yes', 'No', 'Do not know'),key="3")
pre=[hazardexpolvl,option1,option2,option3,option4,option5,option6,q_water,challenlevel,elec_source_renew]
#st.write(pre)
Nn = NearestNeighbors(n_neighbors=3)
Nearest = Pipeline([
    ('preprocessing', preprocessor),
    ('knn', Nn)
])
numcol=['Hazards.Exposure.Level','Potable.Water.Supply.Percent','Adaptation.Challenges.Level','Electricity.Source.Renewable']
#predi=np.array([100.0,'No','Yes','Yes','No','No target','Yes',0.0,3.0,25.0]).reshape(1,10)
#st.write(predi)
predi = np.array(pre).reshape(1,10)
pred = pd.DataFrame(data=predi,columns=col_to_use)
#pred.iloc[0,:] = [100.0,'No','Yes','Yes','No','No target','Yes',0.0,3.0,25.0]
#print(pred)
#st.write(pred)
#st.dataframe(pred)
pred[numcol] = pred[numcol].convert_objects(convert_numeric=True)
final_voisin = Nearest.fit(X)
pred_scale=Nearest.steps[0][1].transform(pred)
#print(pred_scale)
#st.write(pred_scale)
st.write("### Results")
vul_predict=final_pipe_trained.predict(pred)
st.write("Vulnerability score: ",vul_predict[0])
#Xr_train, Xr_test, yr_train, yr_test = train_test_split(X,readiness, test_size=0.4, random_state=1)
final_piper_trained = final_pipel.fit(Xr_train,yr_train)
read_predict = final_piper_trained.predict(pred)
st.write("Readiness score: ",read_predict[0])
st.write("Resilience score: ", (read_predict[0]*(1-vul_predict[0]))/(read_predict[0]+(1-vul_predict[0])))

st.write("")
st.write('#### Please select the number of neighbors to your city you want to see ? (From 1 to 10)')
nb_voisin = st.slider('', 1, 10, 1)
voisin = final_voisin.steps[1][1].kneighbors(pred_scale,n_neighbors=nb_voisin)
st.write("Number of neighbors selected: ",nb_voisin)
ville_voisine= voisin[1][0]
#st.write(ville_voisine)
zoom_start = 1
m = folium.Map(location=[ 43.3,  5.4],zoom_start=zoom_start)
for i in ville_voisine:

    icity= df.iloc[i]
    lat=icity[55]
    lon=icity[56]
    txt=f'readiness score:{icity[58]} \n vulnerability score:{icity[59]} \n Orga:{icity[3]}'
    print(txt)
    #folium.Marker(
    folium.CircleMarker(

    location=[lat,lon],
    popup=txt,
    #icon=folium.Icon(color=couleur)
    radius=5,
    color="green",
    fill=True,
    fill_color='gray'
    ).add_to(m)

folium_static(m)