import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
df_allSubjects  = pd.read_excel('/Users/yiweiwang/Desktop/INF2-FDS/NSS23_Summary_Registered_Full-time.xlsx',header=3, 
sheet_name='All subjects', engine='openpyxl')
#now try to delete all columns with provider name as a country
country_list = ['UK', 'England', 'Scotland', 'Wales', 'Northern Ireland']
for i in country_list:
    df_allSubjects = df_allSubjects[df_allSubjects['Provider name'] != i]
#clean the data, check if there's NaN under the positivity measure column
df_allSubjects = df_allSubjects.dropna(subset=['Positivity measure (%)'])
russell_group_universities = [
    "University of Birmingham",
    "University of Bristol",
    "University of Cambridge",
    "Cardiff University",
    "Durham University",
    "The University of Edinburgh",
    "University of Exeter",
    "University of Glasgow",
    "Imperial College London",
    "King's College London",
    "University of Leeds",
    "University of Liverpool",
    "The London School of Economics and Political Science (LSE)",
    "University of Manchester",
    "Newcastle University",
    "University of Nottingham",
    "University of Oxford",
    "Queen Mary University of London",
    "Queen's University Belfast",
    "University of Sheffield",
    "University of Southampton",
    "University College London",
    "University of Warwick",
    "University of York"
]
df_allSubjects['Provider name'] = df_allSubjects['Provider name'].replace({
    'The University of Birmingham': 'University of Birmingham',
    'University of Durham': 'Durham University',
    'Imperial College of Science, Technology and Medicine': 'Imperial College London',
    'University of Edinburgh': 'The University of Edinburgh',
    'The University of Leeds': 'University of Leeds',
    'The University of Liverpool': 'University of Liverpool',
    'The London School of Economics and Political Science': 'The London School of Economics and Political Science (LSE)',
    'The University of Manchester': 'University of Manchester',
    'University of Newcastle upon Tyne': 'Newcastle University',
    'University of Nottingham, The': 'University of Nottingham',
    "Queen's University of Belfast": "Queen's University Belfast",
    'The University of Sheffield': 'University of Sheffield',
    'The University of Warwick': 'University of Warwick',
})

#combute the ranking of universities in the all subjects dataset
df_allSubjects = df_allSubjects.groupby('Provider name')['Positivity measure (%)'].mean().reset_index()
df_allSubjects['domestic-satisfaction'] = df_allSubjects['Positivity measure (%)'].rank(method='min',ascending=False)
#now drop all non-russell group universities
df_allSubjects = df_allSubjects[df_allSubjects['Provider name'].isin(russell_group_universities)]
df_allSubjects = df_allSubjects[['Provider name', 'domestic-satisfaction']].reset_index(drop=True)
df_allSubjects['domestic-ranking'] = None
#now add the data from complete university guide by hand
df_allSubjects.at[0, 'domestic-ranking'] = 1
df_allSubjects.at[1, 'domestic-ranking'] = 8
df_allSubjects.at[2, 'domestic-ranking'] = 6
df_allSubjects.at[3, 'domestic-ranking'] = 24
df_allSubjects.at[4, 'domestic-ranking'] = 30
df_allSubjects.at[5, 'domestic-ranking'] = 53
df_allSubjects.at[6, 'domestic-ranking'] = 27
df_allSubjects.at[7, 'domestic-ranking'] = 3
df_allSubjects.at[8, 'domestic-ranking'] = 12
df_allSubjects.at[9, 'domestic-ranking'] = 9
df_allSubjects.at[10, 'domestic-ranking'] = 14
df_allSubjects.at[11, 'domestic-ranking'] = 16
df_allSubjects.at[12, 'domestic-ranking'] = 1
df_allSubjects.at[13, 'domestic-ranking'] = 15
df_allSubjects.at[14, 'domestic-ranking'] = 26
df_allSubjects.at[15, 'domestic-ranking'] = 22
df_allSubjects.at[16, 'domestic-ranking'] = 24
df_allSubjects.at[17, 'domestic-ranking'] = 19
df_allSubjects.at[18, 'domestic-ranking'] = 28
df_allSubjects.at[19, 'domestic-ranking'] = 2
df_allSubjects.at[20, 'domestic-ranking'] = 20
df_allSubjects.at[21, 'domestic-ranking'] = 17
df_allSubjects.at[22, 'domestic-ranking'] = 11
df_allSubjects.at[23, 'domestic-ranking'] = 17
import matplotlib.pyplot as plt

# Create a figure and a set of subplots
fig, ax = plt.subplots()

# Plot the 'domestic-ranking' column on the y-axis
ax.plot(df_allSubjects.index, df_allSubjects['domestic-ranking'], color='tab:blue', label='Domestic Ranking', marker='o')

# Plot the 'Positivity Ranking' column on the same y-axis
ax.plot(df_allSubjects.index, df_allSubjects['domestic-satisfaction'], color='tab:red', label='Positivity Ranking', marker='o')

ax.set_xlabel('Provider name')
ax.set_ylabel('Ranking')

# Hide x-axis labels
plt.xticks([])

# Show the plot
plt.legend()
plt.show()