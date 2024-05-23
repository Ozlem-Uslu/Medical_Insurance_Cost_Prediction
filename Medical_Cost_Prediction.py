############################
#MEDICAL COST PERSONAL
############################

########################################################
#Importieren der wesentlichen Bibliotheken und Metriken
########################################################
#Datenverarbeitungsbibliotheken
import joblib
import pandas as pd
import numpy as np

#Visualisierungsbibliotheken
import matplotlib.pyplot as plt
import seaborn as sns


#Feature Engineering-Bibliotheken
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

#Bibliotheken für maschinelles Lernen
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV

#Anpassungen, um Warnungen zu entfernen und eine bessere Überwachung zu ermöglichen
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

#Datensatz importieren
df = pd.read_csv("Final_Project/insurance.csv")


##################################
#Exploratory Data Analysis(EDA)
##################################

#Daten ansehen und beschreiben
def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(3))
    print("##################### Tail #####################")
    print(dataframe.tail(3))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Nunique #####################")
    print(dataframe.nunique())
check_df(df)

#Duplikate behandeln
df.duplicated().sum()
duplicate_rows_data = df[df.duplicated()]
df = df.drop_duplicates()

df = df.reset_index(drop=True)

##################################################
#Kategorisierung numerischer und kategorialer Variablen
##################################################
def grab_col_names(dataframe, cat_th=10, car_th=20):

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# alle Variablen in numerische Werte umwandeln und ihre Korrelation mit der abhängigen Variable "charges" untersuchen.
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" ]

for col in binary_cols:
    label_encoder(df, col)

df.corr()['charges'].sort_values()

# die Korrelationsbeziehung in der Heatmap
fig = plt.figure(figsize = (6, 4))
data_ploting = df.corr(method= 'pearson')
sns.heatmap(data_ploting, cmap='Reds', linecolor='black', linewidths= 2 )
plt.show()

#############################
#Visualisierung der Daten
#############################
df = pd.read_csv("Final_Project/insurance.csv")

####################
#Zuerst die CHARGES variable untersuchen
####################
#Verteilung der medizinischen Kosten
plt.figure(figsize=(8, 6))
plt.hist(df['charges'], bins=20, edgecolor='black')
plt.xlabel('charges')
plt.ylabel('Frequency')
plt.title('charges Distribution')
plt.show()

#Diese Verteilung ist rechtsschief. Wir können den natürlichen Logarithmus anwenden, um den Normalwert anzunähern.
df["charges"]=np.log1p(df['charges'])

#Gruppiertes Balkendiagramm von sex und charges
plt.figure(figsize=(8, 6))
sns.barplot(data=df, x='sex', y='charges')
plt.xlabel('Gender')
plt.ylabel('Charges')
plt.title('Gender vs. Charges')
plt.show()

############################
#die Variable "smoker" untersuchen
############################
#Kreisdiagramm der Raucher- vs. Nichtraucherverteilung
plt.figure(figsize=(8, 6))
df['smoker'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Smoker vs. Non-Smoker Distribution')
plt.show()

#die Grafik zwischen smoker und charges Variablen
plt.figure(figsize=(8, 6))
sns.barplot(data=df, x='smoker', y='charges')
plt.xlabel('Smoker')
plt.ylabel('Charges')
plt.title('Smoker vs. Charges')
plt.show()
#HINWEIS: Patienten, die rauchen, geben mehr Geld für die Behandlung aus. Allerdings ist die Zahl der Nichtraucherpatienten höher.

#die Situation von smoker nach Geschlecht untersuchen.
plt.figure(figsize=(8, 6))
sns.barplot(data=df, x='smoker', y='charges', hue='sex')
plt.xlabel('smoker')
plt.ylabel('Charges')
plt.title('Smoker vs. Charges')
plt.show()
#HINWEIS: Männliche Raucher zahlen mehr Gesundheitskosten.

#Verteilung der Ausgaben nach Anzahl der Kinder von Rauchern
plt.figure(figsize=(8, 6))
sns.barplot(data=df, x='children', y='charges', hue='smoker')
plt.xlabel('Children')
plt.ylabel('Charges')
plt.title('Children vs. Charges')
plt.show()
#HINWEIS: Wer Kinder hat und raucht, hat höhere Gesundheitskosten

#HINWEIS: Rauchen hat den größten Einfluss auf die medizinischen Kosten, obwohl die Kosten mit zunehmendem Alter,
# Body-Mass-Index und Kindern steigen. Außerdem rauchen Menschen mit Kindern im Allgemeinen weniger.

################################
#Die Variable "age" untersuchen
################################
#Balkendiagramm der Altersverteilung
plt.figure(figsize=(20,12))
ax = sns.countplot(data=df, y="age")
plt.xlabel('Count')
plt.ylabel('Age')
ax.bar_label(ax.containers[0])
plt.title('Age Distribution')
plt.show()
#HINWEIS: Wir haben Patienten unter 20 Jahren in unserem Datensatz. 18 Jahre ist das Mindestalter für Patienten in unserer Gruppe. Das Höchstalter beträgt 64 Jahre.
# Mein persönliches Interesse ist, ob es unter 18-jährigen Patienten Raucher gibt.

#Streudiagramm von age vs. charges
plt.figure(figsize=(8, 6))
plt.scatter(df['age'], df['charges'])
plt.xlabel('Age')
plt.ylabel('Charges')
plt.title('Age vs. Charges')
plt.show()

#Schauen wir uns den Raucherstatus von 18-jährigen Männern und Frauen an.
plt.figure(figsize=(8, 6))
sns.countplot(data=df[(df.age == 18)], x='smoker', hue='sex',palette="rainbow")
plt.xlabel('smoker')
plt.ylabel('Count')
plt.title('The number of smokers and non-smokers (18 years old)')
plt.show()
#HINWEIS: Es gibt mehr Männer unter 18 Jahren als Raucher. Unter den Nichtrauchern ist der Anteil der Frauen höher
# und der Anteil der Nichtraucher insgesamt höher.

# Behandlungskosten für Raucher und Nichtraucher ab 18 Jahren
plt.figure(figsize=(8, 6))
sns.boxplot(data=df[(df.age == 18)], x='smoker', y='charges')
plt.xlabel('Smoker')
plt.ylabel('Age')
plt.title('Box plot for charges 18 years old smokers')
plt.show()

#Behandlungskosten für Raucher und Nichtraucher ab 64 Jahren
plt.figure(figsize=(8, 6))
sns.boxplot(data=df[(df.age == 64)], x='smoker', y='charges')
plt.xlabel('Smoker')
plt.ylabel('Age')
plt.title('Box plot for charges 64 years old smokers')
plt.show()
#HINWEIS: Wie man sieht, geben Raucher selbst im Alter von 18 Jahren deutlich mehr Geld für eine Behandlung aus als Nichtraucher.
#64-jährige Raucher geben dreimal so viel aus wie Nichtraucher. 18-Jährige kosten etwa das Siebenfache.

################################
#Die Variable "bmi" untersuchen
################################
#Histogramm der BMI-Werte
plt.figure(figsize=(8, 6))
plt.hist(df['bmi'], bins=20, edgecolor='black')
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.title('BMI Distribution')
plt.show()

#Klassifizierung des Zusammenhangs zwischen BMI und charges durch smoker
plt.figure(figsize=(10, 6))
plt.scatter(df['bmi'], df['charges'], cmap='coolwarm', alpha=0.7)
plt.colorbar(label='smoker')
plt.xlabel('BMI')
plt.ylabel('Charges')
plt.title('BMI and charges')
plt.grid(True)
plt.show()

################################
#Die Variable "children" untersuchen
################################
#Balkendiagramm der Anzahl der Kinder/Angehörigen
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='children')
plt.xlabel('Number of Children')
plt.ylabel('Count')
plt.title('Number of Children Distribution')
plt.show()
#HINWEIS: Die meisten Patienten haben keine Kinder.

#Untersuchen, ob diejenigen, die Kinder haben, rauchen oder nicht.
plt.figure(figsize=(8, 6))
sns.countplot(data=df[(df.children > 0)], x='smoker', hue='sex')
plt.xlabel('Smoker')
plt.ylabel('Count')
plt.title('Smokers and non-smokers who have children')
plt.show()
#HINWEIS: Die Zahl der Nichtrauchereltern ist überwiegend gut.

################################
#Die Variable "region" untersuchen
################################
#Balkendiagramm der Regionsverteilung
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='region')
plt.xlabel('Region')
plt.ylabel('Count')
plt.title('Region Distribution')
plt.show()

#Gestapeltes Balkendiagramm der Region vs. Gebühren
plt.figure(figsize=(8, 6))
sns.barplot(data=df, x='region', y='charges')
plt.xlabel('Region')
plt.ylabel('Charges')
plt.title('Region vs. Charges')
plt.show()

#Untersuchung der Verteilung der Regionen nach Gesamtkosten Wir haben oben die durchschnittlichen Kosten der Regionen gesehen.
charges = df['charges'].groupby(df["region"]).sum().sort_values(ascending = True)
#HINWEIS: Die höchsten Gesundheitsausgaben werden im Südwesten getätigt, die geringsten im Südwesten.

#Boxplot der Gebühren gruppiert nach Region
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='region', y='charges')
plt.xlabel('Region')
plt.ylabel('Charges')
plt.title('Charges Distribution by Region')
plt.show()

####################################################
#Untersuchen der Zielvariablen mit kategorialen Variablen
####################################################
cat_cols, num_cols, cat_but_car = grab_col_names(df)
def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")
for col in cat_cols:
    target_summary_with_cat(df, "charges", col)


###########################################
#FEATURE ENGINEERING
###########################################

###################################################
# Label Encoding & One-Hot Encoding
###################################################
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and len(df[col].unique()) == 2]
cat_columns = [col for col in df.columns if df[col].dtypes == "O" and len(df[col].unique()) == 4]
for col in binary_cols:
    label_encoder(df, col)
for col in cat_columns:
    label_encoder(df, col)

df.head()

########################################
#StandartScaler
########################################
cat_cols, num_cols, cat_but_car, = grab_col_names(df)

num_cols = [col for col in num_cols if "charges" not in col]
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])


##################################
# MODELLING
##################################
#abhängige und unabhängige Variablen trennen.
y = df["charges"]
X = df.drop(["charges","region"], axis=1)
#Hinweis:Da die Korrelation der Regionsvariablen mit der Zielvariablen gering ist, habe ich die Regionsvariable beim Erstellen
#des Modells nicht einbezogen.

###############################################################
#Log Convertion for charges
##############################################################
y = np.log1p(df['charges'])
X = df.drop(["charges"], axis=1)
y.head()

#Trennung von Train- und Testsätzen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


models = [('LR', LinearRegression()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor(random_state=1)),
          ('RF', RandomForestRegressor(random_state=1)),
          ('GBM', GradientBoostingRegressor(random_state=1)),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor(verbosity=-1, random_state=1))]
for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

#Schauen wir uns die R2-Werte der Modelle an.
for regressor_name, regressor in models:
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    accuracy = round(r2_score(y_test,y_pred),1)*100
    print('{:s} : {:.0f} %'.format(regressor_name, accuracy))
    plt.rcParams["figure.figsize"] = (5,3)
    plt.bar(regressor_name,accuracy)

#############################
#GBM-Modell
#############################
gbm_model = GradientBoostingRegressor(random_state=42).fit(X_train, y_train)
y_pred = gbm_model.predict(X_test) #Zuerst kehren wir den Logarithmus des y-Werts um
y_pred = np.expm1(y_pred)
y_test = np.expm1(y_test)
y_train=np.expm1(y_train)
y = np.expm1(y)

#RMS-Wert
rmse = np.mean(np.sqrt(-cross_val_score(gbm_model, X, y, cv=10, scoring="neg_mean_squared_error")))

###################################
#Hyperparameter-Optimierung
###################################
gbm_model.get_params()
param_grid = {
    'n_estimators': [100, 500, 1000],
    'learning_rate': [0.01, 0.1, 1],
    'max_depth': [3, 4, 5]
}
gbm_gs_best = GridSearchCV(gbm_model,
                            param_grid,
                            cv=10,
                            n_jobs=-1,
                            verbose=-1).fit(X_train, y_train)
gbm_gs_best.best_params_


final_model = gbm_model.set_params(**gbm_gs_best.best_params_).fit(X, y)
rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=5, scoring="neg_mean_squared_error")))

#R2-Punktzahl des Train-sets
y_pred_tr = final_model.predict(X_train)
r2_score(y_train,y_pred_tr)

#R2-Punktzahl des Test-sets
y_pred_tx=final_model.predict(X_test)
r2_score(y_test,y_pred_tx)

#Exportieren Sie das trainierte Modell mit joblib
import joblib
model_filename = "final_model.joblib"
joblib.dump(final_model, model_filename)
print(f'Model saved as {model_filename}')

