import pandas as pd 
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np


df = pd.read_excel('https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx')
column_names = {'X1':'Relative_compactness', 'X2':'Surface_Area', 'X3':'Wall_Area',
                'X4':'Roof_Area', 'X5':'Overall_Height', 'X6':'Orientation', 'X7':'Glazing_Area', 'X8':'Glazing_Area_Distribution',
                'Y1':'Heating_Load','Y2':'Cooling_Load'}
df.rename(columns=column_names,inplace=True)

simple_linear_reg_df = df[['Relative_compactness','Cooling_Load']].sample(15,random_state=2)
sns.regplot(x='Relative_compactness',y='Cooling_Load',data=simple_linear_reg_df)



scaler = MinMaxScaler()
normalised_df = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)
features_df = normalised_df.drop(columns=['Heating_Load', 'Cooling_Load'])
heating_target = normalised_df['Heating_Load']

x_train, x_test, y_train, y_test = train_test_split(features_df, heating_target, test_size=0.3, random_state=1)
linear_model = LinearRegression()
linear_model.fit(x_train,y_train)
predicted_values = linear_model.predict(x_test)

mae = mean_absolute_error(y_test, predicted_values)
rss = np.sum(np.square(y_test - predicted_values))
rmse = np.sqrt(mean_squared_error(y_test, predicted_values))
r2_score = r2_score(y_test, predicted_values)
