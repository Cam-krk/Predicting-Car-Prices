import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd


df = pd.DataFrame({
    'Model': ['A4', 'A6', 'A8', '3 Series', '5 Series'],
    'Horsepower': [150, 200, 250, 180, 220],
    'Price_in_thousands': [25, 30, 45, 28, 36]
})

X = df.drop(['Price_in_thousands'], axis=1)  
y = df['Price_in_thousands']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor()
model.fit(X_train[['Horsepower']], y_train)

st.title('Car Price Prediction Tool')

horsepower = st.number_input('Horsepower', min_value=50, max_value=500, value=150)


model_input = st.selectbox('Model', df['Model'].unique())

if st.button('Predict Price'):
    input_df = pd.DataFrame([[model_input, horsepower]], columns=['Model', 'Horsepower'])
    
    
    prediction = model.predict(input_df[['Horsepower']])
    
    
    st.success(f'The estimated price of the car is: ${prediction[0]:.2f} thousand')

