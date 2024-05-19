import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import plotly.express as px

st.set_page_config(layout="wide")

@st.cache_data
def get_data():
    df = pd.read_csv('insurance.csv')
    return df


st.title("👨‍⚕️💊Medical Insurance Cost Prediction💵")

tab_home, tab_data, tab_charts, tab_model = st.tabs(["Homepage", "Dataset","Charts", "Model"] )


####################TAB HOMEPAGE#####################

columns_1, columns_2 = tab_home.columns(2, gap="large")
columns_1.subheader("**🏥Who is MEDI medical insurance company?**")
columns_1.markdown("* MEDI Health Insurance is one of the leading companies in the sector, known for its innovative approaches and customer-oriented policies in America.")
columns_1.markdown("* MEDI's success lies in its data-driven decision-making and analytics approach. By gaining a deeper understanding of its customers' health habits and history, the company develops strategies based on comprehensive data analysis to understand and effectively manage healthcare costs.")
columns_1.subheader("**🛠️Project Phases**")
columns_1.markdown("* Exploratory Data Analysis")
columns_1.markdown("* Data Preprocessing")
columns_1.markdown("* Feature Engineering")
columns_1.markdown("* Modelling")
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
column_4.subheader("❗ Important Stage in the Project", divider="grey")
column_4.markdown("️To increase the performance of the model, I tried seven different model types by adjusting the hyperparameters and built the model with the :red[GBM] machine learning algorithm that gave the best results.")


#####################TAB CHARTS###############################
#1.Correlation
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

fig1, ax = plt.subplots(figsize=(6,3))
data_ploting = df.corr(method= 'pearson')
ax = sns.heatmap(data_ploting, cmap='Reds', linecolor='black', linewidths= 2)
plt.xticks(size=7)
plt.yticks(size=7)
tab_charts.pyplot(fig1, use_container_width=False)


#2.Graph of variables highly correlated with charges
tab_charts.subheader(":bar_chart: Graph of variables highly correlated with charges ",divider="grey")
selected_column=tab_charts.selectbox("Select a Column:", ["smoker","age", "bmi"])

if selected_column == "smoker":
    df1 = df.groupby(["smoker"]).agg({"charges": "mean"})
    df1.reset_index(inplace=True)
    df1["smoker"] = df1["smoker"].map({0: 'no', 1: 'yes'})
    fig2 = px.bar(df1, x="smoker", y="charges", width=800, height=700)
    tab_charts.plotly_chart(fig2, use_container_width=True)
elif selected_column == "age":
    df2 = df.groupby(["age"]).agg({"charges": "mean"})
    df2.reset_index(inplace=True)
    fig3 = px.bar(df2, x="age", y="charges", width=800, height=700)
    tab_charts.plotly_chart(fig3, use_container_width=True)
elif selected_column == "bmi":
    df3 = df.groupby(["bmi"]).agg({"charges": "mean"})
    df3.reset_index(inplace=True)
    fig4 = px.area(df3, x="bmi", y="charges", width=800, height=700)
    tab_charts.plotly_chart(fig4, use_container_width=True)


#3.Annual fees paid according to age and smoking status
tab_charts.subheader("💲Annual charges paid by age and smoking status", divider="grey")
df["smoker"] = df["smoker"].map({0: 'no', 1:'yes'})
selected_smoker = tab_charts.multiselect(label="Select Smoker", options=df.smoker.unique())
filtered_smoker = df[df.smoker.isin(selected_smoker)]

df4= filtered_smoker.groupby(["age"]).agg({"charges": "mean"})
df4.reset_index(inplace=True)
fig5 = px.histogram(df4, x="age", y="charges",width=800, height=600)
tab_charts.plotly_chart(fig5, use_container_width=True)


#4.Cost amount according to bmi and smoker
tab_charts.subheader("💲Medical insurance cost amount according to BMI and smoker", divider="grey")
df_ = df["smoker"].map({0: 'No', 1:'Yes'})
selected_smoker1 = tab_charts.multiselect(label="Smoker", options=df.smoker.unique())
filtered_smoker = df[df.smoker.isin(selected_smoker1)]
fig6=px.scatter(filtered_smoker, x="bmi", y="charges", color="smoker", width=800, height=600)
tab_charts.plotly_chart(fig6, use_container_width=True)

#5.Charges by gender and number of children
tab_charts.subheader("💲Charges by gender and number of children", divider="grey")
selected_children = tab_charts.multiselect(label="Children", options=df.children.unique())
filtered_children = df[df.children.isin(selected_children)]
df6 = filtered_children.groupby(["sex"]).agg({"charges": "mean"})
df6.reset_index(inplace=True)
df6["sex"] = df6["sex"].map({1: 'male', 0:'female'})
fig7 = px.bar(df6, x="sex", y="charges", width=800, height=600)
tab_charts.plotly_chart(fig7, use_container_width=True)

########################TAB MODELLING########################
def get_model():
    model = joblib.load("final_model.joblib")
    return model

model = get_model()

age = tab_model.text_input('Age')
sex = tab_model.number_input('Sex: :red[0: Female], :blue[1: Male]', min_value=0, max_value=1)
bmi = tab_model.text_input('Body Mass Index : :green[weight/(height)²]')
children = tab_model.number_input('Number of Children', min_value=0)
smoker = tab_model.number_input('Smoker: :green[0: No], :red[1: Yes]', min_value=0, max_value=1)
#region = tab_model.number_input('Region of Living: :blue[0: NorthEast], :red[1: NorthWest], :green[2: SouthEast], 3: SouthWest',
                               # min_value=0, max_value=3, step=1)

user_input = pd.DataFrame({"age":age, "sex":sex, "bmi":bmi, "children":children, "smoker":smoker}, index=[0])

if tab_model.button(':rainbow[Predict!]'):
    prediction = model.predict(user_input)
    tab_model.success(f"Predicted Medical Insurance Cost:{round(prediction[0], 2) }")
    tab_model.balloons()

























