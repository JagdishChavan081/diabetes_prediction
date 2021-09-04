# Description: This program detets if someone has diabets using machine learning and Python !


#Import the Libraries
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

#create a title and sub-title

st.write("""
# Diabetes Detection
Detect if someone has diabetes using machine learning and python !
 """)

 #Open and display an Image
image = Image.open('/home/jc/project/diabetes_Ml/5_resources/download.jpeg')

st.image(image,caption='Diabetes using ML',use_column_width=True)


 #get The data
df = pd.read_csv('/home/jc/project/diabetes_Ml/3_dset/diabetes.csv')

#set the subheader on web app
st.subheader('Data Information: ')

#show the data as a table
st.dataframe(df)

#show statictics on data
st.write(df.describe())

#show the data as a chart
st.bar_chart(df)

#Split the data into independent 'x' and dependent 'y' variables

x = df.iloc[:,0:8].values
y=df.iloc[:,-1].values

#split the dataset into 75% training and 25% testing

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25, random_state=0)


# get the feature input from the user
def get_user_input():
    Pregnancies = st.sidebar.slider('Pregnancies',0,17,3)
    Glucose= st.sidebar.slider('Glucose',0,199,117)
    BloodPressure = st.sidebar.slider('BloodPresure',0,122,72)
    SkinThickness = st.sidebar.slider('SkinThickness',0,99,23)
    Insulin = st.sidebar.slider('Insulin',0.0,846.0,30.5)
    BMI = st.sidebar.slider('BMI',0.0,67.1,32.0)
    DiabetesPedigreeFunction = st.sidebar.slider('DiabetesPedigreeFunction',0.078,2.42,0.3725)
    Age = st.sidebar.slider('Age',21,81,29)


    #store a dictonary into a variable
    user_data = {"Preagnancies":Pregnancies,
            "Glucose":Glucose,
            "BloodPressure":BloodPressure,
            "SkinThickness":SkinThickness,
            "Insulin":Insulin,
            "BMI":BMI,
            "DiabetesPedigreeFunction":DiabetesPedigreeFunction,
            "Age":Age
            }


    #transform the data into data frame
    features = pd.DataFrame(user_data, index=[0])
    return features

#Store the user inpt into a variable
user_input=get_user_input()

#set a subheader and display the user input
st.subheader('User Input')
st.write(user_input)

#create and train the model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(x_train, y_train)

#show the models metrics
st.subheader('Model Test Accuracy Score:')
st.write(str(accuracy_score(y_test, RandomForestClassifier.predict(x_test)) *100)+"%")


#Store the model prediction in a variable
prediction = RandomForestClassifier.predict(user_input)

#set a subheader and display the classification
st.subheader('Classification')
st.write(prediction)
