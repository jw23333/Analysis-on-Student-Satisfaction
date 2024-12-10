import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import subjectsCategory
df_CAH1 = pd.read_excel('/Users/yiweiwang/Desktop/INF2-FDS/finalProject/NSS23_Summary_Registered_Full-time.xlsx',header=3,
sheet_name='CAH_level_1', engine='openpyxl')
df_CAH2 = pd.read_excel('/Users/yiweiwang/Desktop/INF2-FDS/finalProject/NSS23_Summary_Registered_Full-time.xlsx',header=3,
sheet_name='CAH_level_2', engine='openpyxl')
df_CAH3 = pd.read_excel('/Users/yiweiwang/Desktop/INF2-FDS/finalProject/NSS23_Summary_Registered_Full-time.xlsx',header=3,
sheet_name='CAH_level_3', engine='openpyxl')
#delete providers with country names
country_list = ['UK', 'England', 'Scotland', 'Wales', 'Northern Ireland']
for i in country_list:
    df_CAH1 = df_CAH1[df_CAH1['Provider name'] != i]
    df_CAH2 = df_CAH2[df_CAH2['Provider name'] != i]
    df_CAH3 = df_CAH3[df_CAH3['Provider name'] != i]
#clean the data, check if there's NaN under the positivity measure column
df_CAH1 = df_CAH1.dropna(subset=['Positivity measure (%)'])
df_CAH2 = df_CAH2.dropna(subset=['Positivity measure (%)'])
df_CAH3 = df_CAH3.dropna(subset=['Positivity measure (%)'])
#now group by subjects, calculate the mean of positivity measure for the same subject
df_grouped_CAH1 = df_CAH1.groupby(['Subject'])[['Positivity measure (%)', 'Benchmark (%)']].mean().reset_index()
df_grouped_CAH2 = df_CAH2.groupby(['Subject'])[['Positivity measure (%)', 'Benchmark (%)']].mean().reset_index()
df_grouped_CAH3 = df_CAH3.groupby(['Subject'])[['Positivity measure (%)', 'Benchmark (%)']].mean().reset_index()
#now concatenate the three dataframes
df_grouped = pd.concat([df_grouped_CAH1, df_grouped_CAH2, df_grouped_CAH3])
df_final = df_grouped.groupby('Subject')[['Positivity measure (%)', 'Benchmark (%)']].mean().reset_index()
df_final['Evaluation'] = df_final[['Positivity measure (%)', 'Benchmark (%)']].mean(axis=1)
#now sort the dataframe by evaluation
df_final = df_final.sort_values(by='Evaluation', ascending=False)
#now plot the top 20 and bottom 20 subjects
df_final_top = df_final.head(20)
df_final_bottom = df_final.tail(20)
plt.figure(figsize=(30, 30))
sns.barplot(y='Subject', x='Evaluation', data=df_final_top, color='skyblue', orient='h')
plt.title('Top 20 subjects')
plt.subplots_adjust(left=0.3)
plt.show()  

plt.figure(figsize=(30, 30))
sns.barplot(y='Subject', x='Evaluation', data=df_final_bottom, color='red', orient='h')
plt.title('Bottom 20 subjects')
plt.subplots_adjust(left=0.3)
plt.show()  