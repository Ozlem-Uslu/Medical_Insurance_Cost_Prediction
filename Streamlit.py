import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(layout="wide")   #görüntünün tüm ekrani kaplamasi icin

@st.cache_data  #veriyi al ön bellege kaydet tekrar calistima demek oluyor
def get_data():
    df = pd.read_csv('Final_Project/insurance.csv')
    return df


#heryerde gözükmesini istedigin seyleri st ile yaz. basligin ve tablarin her yerde görünmesini istiyorum oyüzden st ile yazariz.
st.title("👨‍⚕️💊Medical Insurance Cost Prediction💵") #basligin tüm sayfalarda görünmesi icin en basta tutuyorum

tab_home, tab_data, tab_charts, tab_model, tab_ins = st.tabs(["Homepage", "Dataset","Charts", "Model", "Insight"] )


####################TAB HOMEPAGE#####################

columns_1, columns_2 = tab_home.columns(2, gap="large") #(kolonlari bölebiliriz .columns([1,2]) burada sol tarafa 1 brm saga 2 brm verir.
columns_1.subheader("**Overview of the Project**")
columns_1.markdown("* Business Problem ")
columns_1.markdown ("* Dataset Story")
columns_1.markdown("* Exploratory Data Analysis(EDA)")
columns_1.markdown("* Modelling")
columns_1.markdown("* Business Recommendations")
columns_2.image("affordable-health-care.webp")

####################TAB DATASET######################

column_3, column_4 = tab_data.columns(2, gap="large")
column_3.subheader(" ❓ Business Problem", divider="grey")
column_3.markdown("MEDI health insurance company wants to more accurately predict customers' health insurance costs to offer customers more affordable prices and improve the company's risk management.")

column_3.subheader("🎯Project goal",divider="grey" )
column_3.markdown(" Creating a machine learning model that predicts a customer's likely health insurance cost, using customer profile and other demographic information.")
#column_3.image("Final_Project/22818296.jpg", width=400)

column_3.subheader(" 📁 Dataset Story", divider="grey")
column_3.markdown("* **1337 Obsevation**")
column_3.markdown("* **7 Variables**")
column_3.markdown("* **4 Categorical Variables**")
column_3.markdown("* **3 Numerical Variables**")

df = get_data()
column_4.dataframe(df,width=800)
column_4.subheader("Important Stage in the Project", divider="grey")
column_4.text("gfnhjh")

#####################TAB CHARTS###############################

#1.Graphs of variables
tab_charts.subheader(":bar_chart: Variables",divider="grey")
selected_column=tab_charts.selectbox("Select a Column:", df.columns)

if selected_column=="region" or selected_column=="children":
    fig1, ax =plt.subplots(1, 1, figsize=(4, 2))
    sns.countplot(data=df, x=selected_column,width=0.8)
    tab_charts.pyplot(fig1,use_container_width=False)
elif selected_column=="age" or selected_column=="bmi" or selected_column=="charges":
    fig2, ax = plt.subplots(figsize=(3, 2))
    sns.histplot(df, edgecolor='black', x=selected_column)
    tab_charts.pyplot(fig2,use_container_width=False)
elif selected_column=="sex":
    fig3, ax = plt.subplots(figsize=(3, 3))
    df["sex"].value_counts().plot.pie(autopct='%1.1f%%')
    tab_charts.pyplot(fig3, use_container_width=False)
elif selected_column=="smoker":
    fig4, ax = plt.subplots(figsize=(3, 3))
    df["smoker"].value_counts().plot.pie(autopct='%1.1f%%')
    tab_charts.pyplot(fig4, use_container_width=False)

#2.Correlation
tab_charts.subheader("📈 Correlation of Variables with Target Variable", divider="grey")

from sklearn.preprocessing import LabelEncoder
    # sex
le = LabelEncoder()
le.fit(df.sex.drop_duplicates())
df.sex = le.transform(df.sex)
    # smoker or not
le.fit(df.smoker.drop_duplicates())
df.smoker = le.transform(df.smoker)
    # region
le.fit(df.region.drop_duplicates())
df.region = le.transform(df.region)

fig1, ax = plt.subplots(figsize=(4,4))
data_ploting = df.corr(method= 'pearson')
ax = sns.heatmap(data_ploting, cmap='Reds', linecolor='black', linewidths= 2 )
plt.xticks(size=7)
plt.yticks(size=7)
tab_charts.pyplot(fig1, use_container_width=False)

#3.Annual fees paid according to age and smoking status
tab_charts.subheader("💲Annual insurance fees based on age and smoking status", divider="grey")
selected_smoker = tab_charts.multiselect(label="Select Smoker", options=df.smoker.unique())
filtered_smoker = df[df.smoker.isin(selected_smoker)]

import plotly.express as px
fig2 = px.bar(filtered_smoker, x="age", y="charges", color="smoker", width=800, height=700)
tab_charts.plotly_chart(fig2, use_container_width=True)

#4.Cost amount according to bmi and smoker
tab_charts.subheader("💲Cost amount according to bmi and smoker", divider="grey")
fig3 = tab_charts.scatter_chart(df, x='bmi', y='charges', color="smoker",width=800, height=700)


########################TAB MODELLING########################
def get_model():
    model = joblib.load("final_model.joblib")
    return model

model = get_model()

age = tab_model.text_input('Age')
sex = tab_model.text_input('Sex: 0: Female, 1: Male')
bmi = tab_model.text_input('Body Mass Index')
children = tab_model.number_input('Number of Children', min_value=0, max_value=5)
smoker = tab_model.text_input('Smoker: 0: No, 1: Yes')
region = tab_model.text_input('Region of Living: 0: NorthEast, 1: NorthWest, 2: SouthEast, 3: SouthWest')

user_input = pd.DataFrame({"age":age, "sex":sex, "bmi":bmi, "children":children, "smoker":smoker, "region":region}, index=[0])

if tab_model.button(':rainbow[Predict!]'):  # butonun calismasi icin if
    prediction = model.predict(user_input)
    tab_model.success(f"Predicted Medical Insurance Cost:{round(prediction[0], 2) }")
    tab_model.balloons()

























