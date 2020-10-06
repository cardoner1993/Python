%matplot inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
###Exploration analisis.

train = pd.read_csv("train.csv",sep=",")
train.dtypes
fig, ax = plt.subplots(figsize=(8,6))
objects = train.select_dtypes(include='object').columns

for col in objects:
    train[col] = train[col].astype('category')
    
##dividir las variables explicativas entre los indicadores de datos meteorologicos y de 
##distribucion de probabilidad.
fig, ax = plt.subplots(figsize=(8,6))
train.groupby('Pclass')['Survived'].plot(kind='kde', ax=ax)

##igual que ggplot hasta que no aplicas alguna sumarizacion a los datos no se muestran.   

fig, ax = plt.subplots(figsize=(8,6))
train.groupby(['Sex']).mean()['Age'].plot(ax=ax,kind='bar')

## prueva dos 
fig, ax = plt.subplots(figsize=(8,6))
train.groupby(['Survived','Sex']).mean()['Age'].plot(ax=ax,kind='bar')


fig, ax = plt.subplots(figsize=(8,6))
train.groupby(['Sex']).mean()['Age'].plot(,ax=ax,kind='bar')
#demanar oriol que enten python quan fem .mean()[variable].

### Grouping and scatterplot of other x variables of the grouping previous.
groups = train.groupby('Sex')
fig, ax = plt.subplots()
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    ax.plot(group.Age, group.Fare, marker='o', linestyle='', ms=12, label=name)
ax.legend()

plt.show()
