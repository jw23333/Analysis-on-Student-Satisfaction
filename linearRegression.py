import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import subjectsCategory
import time
start_time = time.time()
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
#now only keep provider name, level of study, subject, question, positivity measure as columns
df_CAH1 = df_CAH1[['Provider name', 'Level of study', 'Subject', 'Question', 'Positivity measure (%)']]
df_CAH2 = df_CAH2[['Provider name', 'Level of study', 'Subject', 'Question', 'Positivity measure (%)']]
df_CAH3 = df_CAH3[['Provider name', 'Level of study', 'Subject', 'Question', 'Positivity measure (%)']]
#now for each df, group by provider name, level of study, subject, and calculate the mean of positivity measure
df_CAH1 = df_CAH1.groupby(['Provider name', 'Level of study', 'Subject'])['Positivity measure (%)'].mean().reset_index()
df_CAH2 = df_CAH2.groupby(['Provider name', 'Level of study', 'Subject'])['Positivity measure (%)'].mean().reset_index()
df_CAH3 = df_CAH3.groupby(['Provider name', 'Level of study', 'Subject'])['Positivity measure (%)'].mean().reset_index()
df_concat = pd.concat([df_CAH1, df_CAH2, df_CAH3])
#now give each provider a unique number for the linear regression
df_concat = pd.get_dummies(df_concat, columns=['Provider name'])
#unique_subjects = df_concat['Subject'].unique()
#print(unique_subjects)
stem_list = subjectsCategory.stem_subjects
#now for the subjects in the concatenated dataframe, label them as 1 if they are stem, 0 if they are humanities
df_concat['Is_STEM'] = df_concat['Subject'].apply(lambda x: 1 if x in stem_list else 0)
#before doing detailed regression, check if the average positivity measure is different between stem and humanities subjects
df_concat_preview = df_concat.groupby('Is_STEM')['Positivity measure (%)'].mean().reset_index()
print(df_concat_preview)
#now do one hot encoding for the level of study
df_concat = pd.get_dummies(df_concat, columns=['Level of study'])#now Level of study column is dropped
#convert boolean to int
#df_concat = df_concat.astype(int)
#now drop the unnecessary columns
x = df_concat.drop(columns=['Positivity measure (%)', 'Subject'])
x = x.astype('float64')
y = df_concat['Positivity measure (%)']
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
print(model.summary())
end_time = time.time()
print('time:', end_time - start_time)
# #results:
# yiweiwang@YiweideMacBook-Pro finalProject % /usr/local/bin/python3 /Users/yiweiwang/Desktop/INF2-FDS/finalProject/linearRegression
# .py
#    Is_STEM  Positivity measure (%)
# 0        0               81.356808
# 1        1               79.666073
#                               OLS Regression Results                              
# ==================================================================================
# Dep. Variable:     Positivity measure (%)   R-squared:                       0.246
# Model:                                OLS   Adj. R-squared:                  0.235
# Method:                     Least Squares   F-statistic:                     23.16
# Date:                    Wed, 20 Mar 2024   Prob (F-statistic):               0.00
# Time:                            14:14:13   Log-Likelihood:                -78980.
# No. Observations:                   23667   AIC:                         1.586e+05
# Df Residuals:                       23338   BIC:                         1.613e+05
# Df Model:                             328                                         
# Covariance Type:                nonrobust                                         
# =================================================================================================================================================
#                                                                                     coef    std err          t      P>|t|      [0.025      0.975]
# -------------------------------------------------------------------------------------------------------------------------------------------------
# const                                                                         -2.541e+12   3.08e+12     -0.825      0.410   -8.58e+12     3.5e+12
# Provider name_ACM Guildford Limited                                            7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_AECC University College                                          7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Abertay University                                               7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Aberystwyth University                                           7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Academy of Live Technology Ltd                                   7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Activate Learning                                                7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Amity Global Education Ltd                                       7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Anglia Ruskin University Higher Education Corporation            7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Architectural Association (Incorporated)                         7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Arden University Limited                                         7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Arts Educational Schools(The)                                    7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Arts University Bournemouth, the                                 7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Arts University Plymouth                                         7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Askham Bryan College                                             7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Aston University                                                 7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_BIMM University Limited                                          7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_BPP University Limited                                           7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Bangor University                                                7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Barnet & Southgate College                                       7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Barnsley College                                                 7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Bath Spa University                                              7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Belfast Metropolitan College                                     7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Birkbeck College                                                 7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Birmingham City University                                       7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Bishop Auckland College                                          7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Bishop Burton College                                            7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Bishop Grosseteste University                                    7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Blackburn College                                                7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Blackpool and the Fylde College                                  7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Bloomsbury Institute Limited                                     7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Boston College                                                   7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Bournemouth University                                           7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Bournemouth and Poole College, The                               7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Bradford College                                                 7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Bridgwater and Taunton College                                   7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Brighton and Sussex Medical School                               7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_British Academy of Jewellery Limited                             7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Brockenhurst College                                             7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Brunel University London                                         7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Buckinghamshire New University                                   7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Burnley College                                                  7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Bury College                                                     7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Calderdale College                                               7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Cambridge Arts & Sciences Limited                                7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Canterbury Christ Church University                              7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Cardiff Metropolitan University                                  7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Cardiff University                                               7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Central School of Ballet Charitable Trust Limited(the)           7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Chichester College Group                                         7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Christ the Redeemer College                                      7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_City College Norwich                                             7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_City College Plymouth                                            7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_City and Guilds of London Art School Limited                     7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_City of Bristol College                                          7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_City of Sunderland College                                       7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_City, University of London                                       7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Colchester Institute                                             7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Contemporary Dance Trust Limited                                 7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Cornwall College                                                 7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Court Theatre Training Company Ltd                               7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Courtauld Institute of Art                                       7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Coventry University                                              7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Croydon College                                                  7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_DCG                                                              7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_DN Colleges Group                                                7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_David Game College Ltd                                           7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_De Montfort University                                           7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_East Sussex College Group                                        7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Edge Hill University                                             7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Edinburgh Napier University                                      7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Elim Foursquare Gospel Alliance                                  7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Empire College London Limited                                    7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Falmouth University                                              7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Farnborough College of Technology                                7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Futureworks Training Limited                                     7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Glasgow Caledonian University                                    7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Glasgow School of Art                                            7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Global Banking School Limited                                    7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Goldsmiths' College                                              7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Grŵp Colegau NPTC Group of Colleges                              7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Grŵp Llandrillo Menai                                            7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Guildhall School of Music & Drama                                7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_HULT International Business School Ltd                           7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Harper Adams University                                          7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Hartpury University                                              7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Havant and South Downs College                                   7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Heart of Worcestershire College                                  7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Heart of Yorkshire Education Group                               7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Hereford College of Arts                                         7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Heriot-Watt University                                           7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Hertford Regional College                                        7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Holy Cross College                                               7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Hopwood Hall College                                             7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Hugh Baird College                                               7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Hull College                                                     7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Hull and York Medical School                                     7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Hy Education Limited                                             7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_ICMP Management Limited                                          7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_ICON College of Technology and Management Ltd                    7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Imperial College of Science, Technology and Medicine             7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Inter-ED UK Limited                                              7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Istituto Marangoni Limited                                       7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_King's College London                                            7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Kingston University                                              7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_LIBF Limited                                                     7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_LTE Group                                                        7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Lamda Limited                                                    7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Le Cordon Bleu Limited                                           7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Leeds Arts University                                            7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Leeds Beckett University                                         7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Leeds Conservatoire                                              7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Leeds Trinity University                                         7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Leicester College                                                7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Lincoln College                                                  7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Liverpool Hope University                                        7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Liverpool John Moores University                                 7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_London Bridge Business Academy Limited                           7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_London Metropolitan University                                   7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_London School of Commerce & IT Limited                           7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_London School of Management Education Limited                    7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_London School of Theology                                        7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_London South Bank University                                     7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_London South East Colleges                                       7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_London Studio Centre Limited                                     7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Loughborough College                                             7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Loughborough University                                          7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Luminate Education Group                                         7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Manchester Metropolitan University                               7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Matrix College of Counselling and Psychotherapy Ltd              7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Met Film School Limited                                          7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Middlesbrough College                                            7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Middlesex University                                             7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Mont Rose College of Management and Sciences Limited             7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Moorlands College                                                7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Moulton College                                                  7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Mountview Academy of Theatre Arts Limited                        7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Myerscough College                                               7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_NCG                                                              7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_National Centre for Circus Arts                                  7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Nazarene Theological College                                     7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Nelson College London Limited                                    7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Nelson and Colne College                                         7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_New College Durham                                               7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_New College Swindon                                              7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Newman University                                                7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Norland College Limited                                          7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_North East Surrey College of Technology (NESCOT)                 7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_North Hertfordshire College                                      7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_North Warwickshire and South Leicestershire College              7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_North West Regional College                                      7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Northeastern University - London                                 7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Northern College of Acupuncture                                  7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Northern Regional College                                        7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Northern School of Contemporary Dance                            7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Norwich University of the Arts                                   7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Nottingham College                                               7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Nottingham Trent University                                      7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Oxford Brookes University                                        7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Paris Dauphine International                                     7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Pearson College Limited                                          7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Peter Symonds College                                            7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Petroc                                                           7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Plumpton College                                                 7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Point Blank Limited                                              7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Preston College                                                  7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Queen Margaret University, Edinburgh                             7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Queen Mary University of London                                  7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Queen's University of Belfast                                    7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_RNN Group                                                        7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_RTC Education Ltd                                                7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Raindance Educational Services Limited                           7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Rambert School of Ballet and Contemporary Dance                  7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Ravensbourne University London                                   7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Reaseheath College                                               7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Regent's University London Limited                               7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Results Consortium Limited                                       7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Richmond, the American International University in London, Inc.  7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Robert Gordon University                                         7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Roehampton University                                            7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Rose Bruford College of Theatre and Performance                  7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Royal Academy of Dance                                           7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Royal Academy of Dramatic Art                                    7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Royal College of Music                                           7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Royal Conservatoire of Scotland                                  7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Royal Holloway and Bedford New College                           7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Royal Northern College of Music                                  7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Royal Welsh College of Music and Drama                           7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_SAE Education Limited                                            7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_SRUC                                                             7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_School of Oriental and African Studies                           7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Sheffield College, The                                           7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Sheffield Hallam University                                      7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Solent University, Southampton                                   7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Solihull College and University Centre                           7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_South Devon College                                              7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_South Eastern Regional College                                   7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_South Essex College of Further and Higher Education              7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_South Gloucestershire and Stroud College                         7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_South Thames Colleges Group                                      7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_South West College                                               7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Southern Regional College                                        7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Sparsholt College                                                7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_St Mary's University College                                     7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_St Mary's University, Twickenham                                 7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_St Mellitus College Trust                                        7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_St. George's Hospital Medical School                             7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Staffordshire University                                         7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Stranmillis University College                                   7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Swansea University                                               7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_TEC Partnership                                                  7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Teesside University                                              7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Telford College                                                  7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_The Cambridge Theological Federation                             7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_The Chicken Shed Theatre Trust                                   7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_The City of Liverpool College                                    7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_The College of Health Ltd                                        7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_The College of Integrated Chinese Medicine                       7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_The Edward James Foundation Limited                              7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_The Film Education Training Trust Limited                        7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_The Liverpool Institute for Performing Arts                      7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_The London School of Economics and Political Science             7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_The Markfield Institute of Higher Education                      7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_The Metanoia Institute                                           7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_The Northern School of Art                                       7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_The Oldham College                                               7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_The Royal Academy of Music                                       7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_The Royal Agricultural University                                7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_The Royal Central School of Speech and Drama                     7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_The Royal Veterinary College                                     7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_The Salvation Army                                               7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_The Sherwood Psychotherapy Training Institute Limited            7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_The Trafford College Group                                       7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_The University of Bath                                           7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_The University of Birmingham                                     7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_The University of Bolton                                         7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_The University of Bradford                                       7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_The University of Buckingham                                     7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_The University of Chichester                                     7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_The University of Cumbria                                        7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_The University of East Anglia                                    7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_The University of Essex                                          7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_The University of Huddersfield                                   7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_The University of Hull                                           7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_The University of Kent                                           7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_The University of Lancaster                                      7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_The University of Law Limited                                    7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_The University of Leeds                                          7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_The University of Leicester                                      7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_The University of Liverpool                                      7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_The University of Manchester                                     7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_The University of Reading                                        7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_The University of Sheffield                                      7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_The University of Surrey                                         7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_The University of Warwick                                        7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_The University of West London                                    7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_The University of Westminster                                    7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_The WKCIC Group                                                  7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Trinity College (Bristol) Limited                                7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Trinity Laban Conservatoire of Music and Dance                   7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Truro and Penwith College                                        7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Tyne Coast College                                               7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_UCK Limited                                                      7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Unified Seevic Palmer's College                                  7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University Centre Peterborough                                   7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University College Birmingham                                    7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University College London                                        7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University College of Estate Management                          7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University College of Osteopathy (The)                           7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University for the Creative Arts                                 7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University of Aberdeen                                           7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University of Bedfordshire                                       7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University of Brighton                                           7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University of Bristol                                            7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University of Cambridge                                          7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University of Central Lancashire                                 7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University of Chester                                            7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University of Derby                                              7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University of Dundee                                             7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University of Durham                                             7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University of East London                                        7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University of Edinburgh                                          7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University of Exeter                                             7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University of Glasgow                                            7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University of Gloucestershire                                    7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University of Greenwich                                          7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University of Hertfordshire                                      7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University of Keele                                              7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University of Lincoln                                            7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University of Newcastle upon Tyne                                7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University of Northampton, The                                   7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University of Northumbria at Newcastle                           7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University of Nottingham, The                                    7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University of Oxford                                             7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University of Plymouth                                           7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University of Portsmouth                                         7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University of Salford, The                                       7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University of South Wales                                        7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University of Southampton                                        7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University of St Andrews                                         7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University of St Mark & St John                                  7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University of Stirling                                           7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University of Strathclyde                                        7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University of Suffolk                                            7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University of Sunderland                                         7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University of Sussex                                             7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University of Ulster                                             7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University of Wales Trinity Saint David                          7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University of Winchester                                         7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University of Wolverhampton                                      7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University of Worcester                                          7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University of York                                               7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University of the Arts, London                                   7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University of the Highlands and Islands                          7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University of the West of England, Bristol                       7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_University of the West of Scotland                               7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Warwickshire College                                             7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_West Herts College                                               7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Weston College of Further and Higher Education                   7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Weymouth College                                                 7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Wigan and Leigh College                                          7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Wirral Metropolitan College                                      7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Wrexham Glyndŵr University                                       7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_Writtle University College                                       7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_York College                                                     7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Provider name_York St John University                                          7.896e+10   1.35e+11      0.586      0.558   -1.85e+11    3.43e+11
# Is_STEM                                                                          -1.6282      0.094    -17.356      0.000      -1.812      -1.444
# Level of study_All undergraduates                                              2.462e+12   3.08e+12      0.798      0.425   -3.58e+12    8.51e+12
# Level of study_First degree                                                    2.462e+12   3.08e+12      0.798      0.425   -3.58e+12    8.51e+12
# Level of study_Other undergraduate                                             2.462e+12   3.08e+12      0.798      0.425   -3.58e+12    8.51e+12
# Level of study_Undergraduate with postgraduate component                       2.462e+12   3.08e+12      0.798      0.425   -3.58e+12    8.51e+12
# ==============================================================================
# Omnibus:                     2664.678   Durbin-Watson:                   2.109
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):             6337.233
# Skew:                          -0.670   Prob(JB):                         0.00
# Kurtosis:                       5.152   Cond. No.                     2.03e+14
# ==============================================================================

# Notes:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
# [2] The smallest eigenvalue is 9.89e-25. This might indicate that there are
# strong multicollinearity problems or that the design matrix is singular.