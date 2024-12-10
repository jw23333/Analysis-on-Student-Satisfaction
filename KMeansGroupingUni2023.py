import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
df_allSubjects = pd.read_excel('/Users/yiweiwang/Desktop/INF2-FDS/finalProject/NSS23_Summary_Registered_Full-time.xlsx',header=3, 
sheet_name='All subjects', engine='openpyxl')
#now try to delete all columns with provider name as a country
country_list = ['UK', 'England', 'Scotland', 'Wales', 'Northern Ireland']
for i in country_list:
    df_allSubjects = df_allSubjects[df_allSubjects['Provider name'] != i]
#clean the data, check if there's NaN under the positivity measure column
df_allSubjects = df_allSubjects.dropna(subset=['Positivity measure (%)'])
#print(df_allSubjects['Positivity measure (%)'].isnull().sum())
# Group the data by 'Provider name' and 'Question', and compute the mean of the 'Positivity measure (%)' and 'Benchmark (%)' values for each group
df_grouped = df_allSubjects.groupby(['Provider name', 'Question'])[['Positivity measure (%)', 'Benchmark (%)']].mean().reset_index()
# Pivot the data so that each row is a university, each column is a question, and the values are the mean positivity measures
df_pivot_positivity = df_grouped.pivot(index='Provider name', columns='Question', values='Positivity measure (%)')
# Pivot the data so that each row is a university, each column is a question, and the values are the mean benchmark values
df_pivot_benchmark = df_grouped.pivot(index='Provider name', columns='Question', values='Benchmark (%)')
# Concatenate the two dataframes along the columns axis
df_features = pd.concat([df_pivot_positivity, df_pivot_benchmark], axis=1)
theme_not_used = ['Theme 1: Teaching on my course', 'Theme 2: Learning opportunities', 'Theme 3: Assessment and feedback', 
'Theme 4: Academic support', 'Theme 5: Organisation and management', 
'Theme 6: Learning resources', 'Theme 7: Student voice']
df_features = df_features.drop(columns=theme_not_used, errors='ignore')
# Drop any columns with NaN values
df_features = df_features.dropna(axis=1, how='any')
#now performs PCA on the features
# Standardize the features
scaler = StandardScaler()
df_features_scaled = scaler.fit_transform(df_features)
# Apply PCA
pca = PCA()
df_features_pca = pca.fit_transform(df_features_scaled)
# Convert the PCA results into a DataFrame
df_features_pca = pd.DataFrame(df_features_pca, index=df_features.index)
# Calculate the cumulative sum of the explained variance ratio
cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
# Find the number of components that explain at least 95% of the variance
n_components_95 = np.where(cumulative_explained_variance >= 0.95)[0][0] + 1
# Get the top contributing features for each PC
top_features_all = set()
for i in range(n_components_95):
    top_features = np.abs(pca.components_[i]).argsort()[::-1][:5]
    top_features_all.update(df_features.columns[top_features])
# Select these columns in df_features
df_features_selected = df_features[list(top_features_all)].copy()
inertias = []
for k in range(1, 20):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df_features_selected)
    inertias.append(kmeans.inertia_)

plt.plot(range(1, 20), inertias, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('The Elbow Method showing the optimal k')
plt.show()
n_clusters = 6
np.random.seed(0)  # Set a random seed
kmeans = KMeans(n_clusters=n_clusters)
df_features_selected['Cluster'] = kmeans.fit_predict(df_features_selected)
# Print the provider names in each cluster
for i in range(n_clusters):
    print(f"Cluster {i}:")
    print(df_features_selected[df_features_selected['Cluster'] == i].index)
#sort the clusters by the mean value of each feature
df_cluster_means = df_features_selected.groupby('Cluster').mean().mean(axis=1).sort_values(ascending=False)
print(df_cluster_means)
top_cluster = df_cluster_means.index[0]
top_cluster_content_2023 = df_features_selected[df_features_selected['Cluster'] == top_cluster].index.tolist()
#now do the bottom cluster
bottom_cluster = df_cluster_means.index[-1]
bottom_cluster_content_2023 = df_features_selected[df_features_selected['Cluster'] == bottom_cluster].index.tolist()
print(top_cluster_content_2023)
# results:
# yiweiwang@YiweideMacBook-Pro finalProject % /usr/local/bin/python3 /Users/yiweiwang/Desktop/INF2-FDS/finalProject/KMeansGroupingUn
# i.py
# Cluster 0:
# Index(['Amity Global Education Ltd', 'Arden University Limited',
#        'Boston College', 'Burnley College', 'Cheshire College South and West',
#        'City of Bristol College', 'Court Theatre Training Company Ltd',
#        'Fareham College', 'Farnborough College of Technology',
#        'Hertford Regional College', 'Hugh Baird College',
#        'Inter-ED UK Limited', 'Kendal College', 'London School of Theology',
#        'London South East Colleges', 'Luminate Education Group',
#        'Moulton College', 'New College Durham', 'Nottingham College',
#        'Raindance Educational Services Limited', 'Sheffield College, The',
#        'South Thames Colleges Group', 'Southport College', 'St Helens College',
#        'TEC Partnership', 'The Cambridge Theological Federation',
#        'The Metanoia Institute', 'University Centre Peterborough',
#        'Weymouth College', 'Wiltshire College and University Centre'],
#       dtype='object', name='Provider name')
# Cluster 1:
# Index(['Barnet & Southgate College', 'Barnsley College',
#        'Bishop Auckland College', 'British Academy of Jewellery Limited',
#        'Brockenhurst College', 'Calderdale College',
#        'Christ the Redeemer College', 'David Game College Ltd',
#        'Empire College London Limited', 'Global Banking School Limited',
#        'Grŵp Colegau NPTC Group of Colleges', 'Havant and South Downs College',
#        'Hy Education Limited', 'Leicester College',
#        'London Bridge Business Academy Limited',
#        'London School of Commerce & IT Limited',
#        'London School of Management Education Limited',
#        'Mont Rose College of Management and Sciences Limited',
#        'Nazarene Theological College', 'Nelson College London Limited',
#        'New College Swindon', 'North Hertfordshire College',
#        'RTC Education Ltd', 'Rambert School of Ballet and Contemporary Dance',
#        'Results Consortium Limited', 'Riverside College',
#        'Southern Regional College', 'Strode College',
#        'The City of Liverpool College', 'The Salvation Army',
#        'The Trafford College Group', 'Trinity College (Bristol) Limited',
#        'UCK Limited', 'Unified Seevic Palmer's College', 'Yeovil College'],
#       dtype='object', name='Provider name')
# Cluster 2:
# Index(['Aberystwyth University', 'Architectural Association (Incorporated)',
#        'Belfast Metropolitan College', 'Bishop Grosseteste University',
#        'Blackburn College', 'Blackpool and the Fylde College',
#        'Bloomsbury Institute Limited', 'Bournemouth and Poole College, The',
#        'Bridgwater and Taunton College', 'Buckinghamshire New University',
#        'Bury College', 'Cambridge Arts & Sciences Limited',
#        'Central School of Ballet Charitable Trust Limited(the)',
#        'City College Norwich', 'City College Plymouth',
#        'City of Sunderland College', 'Coventry University', 'Croydon College',
#        'DCG', 'DN Colleges Group', 'EKC Group', 'East Sussex College Group',
#        'Elim Foursquare Gospel Alliance', 'Exeter College', 'Furness College',
#        'Futureworks Training Limited', 'Grŵp Llandrillo Menai',
#        'HULT International Business School Ltd', 'Hartpury University',
#        'Heart of Worcestershire College', 'Heart of Yorkshire Education Group',
#        'Hereford College of Arts', 'Holy Cross College',
#        'ICON College of Technology and Management Ltd',
#        'Istituto Marangoni Limited', 'Lamda Limited',
#        'Leeds Trinity University', 'London Studio Centre Limited',
#        'Loughborough College',
#        'Matrix College of Counselling and Psychotherapy Ltd',
#        'Middlesbrough College', 'Moorlands College', 'NCG',
#        'National Centre for Circus Arts', 'Newman University',
#        'Norland College Limited',
#        'North East Surrey College of Technology (NESCOT)',
#        'North Warwickshire and South Leicestershire College',
#        'North West Regional College', 'Northern College of Acupuncture',
#        'Northern Regional College', 'Northern School of Contemporary Dance',
#        'Petroc', 'RNN Group', 'Regent's University London Limited',
#        'Shrewsbury Colleges Group', 'Solihull College and University Centre',
#        'South Devon College', 'South Eastern Regional College',
#        'South Essex College of Further and Higher Education',
#        'South Gloucestershire and Stroud College', 'South West College',
#        'St Mary's University, Twickenham', 'St Mellitus College Trust',
#        'Teesside University', 'The Chicken Shed Theatre Trust',
#        'The Markfield Institute of Higher Education',
#        'The Northern School of Art', 'The Oldham College', 'The SMB Group',
#        'The Sherwood Psychotherapy Training Institute Limited',
#        'The University of Bath', 'The University of Bolton',
#        'The University of Reading', 'The University of Surrey',
#        'The University of West London', 'Truro and Penwith College',
#        'University College Birmingham', 'University of St Andrews',
#        'University of Suffolk', 'University of Sunderland',
#        'University of Ulster', 'University of Wales Trinity Saint David',
#        'West Herts College', 'Wigan and Leigh College',
#        'Wrexham Glyndŵr University', 'Writtle University College'],
#       dtype='object', name='Provider name')
# Cluster 3:
# Index(['Abertay University', 'Activate Learning',
#        'Anglia Ruskin University Higher Education Corporation',
#        'Arts Educational Schools(The)', 'Arts University Bournemouth, the',
#        'Arts University Plymouth', 'BPP University Limited',
#        'Bangor University', 'Bath Spa University',
#        'Birmingham City University',
#        ...
#        'University of Winchester', 'University of Wolverhampton',
#        'University of Worcester', 'University of the Arts, London',
#        'University of the Highlands and Islands',
#        'University of the West of Scotland',
#        'Weston College of Further and Higher Education',
#        'Wirral Metropolitan College', 'York College',
#        'York St John University'],
#       dtype='object', name='Provider name', length=104)
# Cluster 4:
# Index(['ACM Guildford Limited', 'AECC University College',
#        'Academy of Live Technology Ltd', 'Askham Bryan College',
#        'BIMM University Limited', 'Cardiff University',
#        'Gloucestershire College', 'Goldsmiths' College',
#        'Hull and York Medical School', 'LTE Group', 'Le Cordon Bleu Limited',
#        'Met Film School Limited', 'Pearson College Limited',
#        'Peter Symonds College', 'Plumpton College',
#        'Ravensbourne University London',
#        'School of Oriental and African Studies',
#        'St. George's Hospital Medical School', 'Telford College',
#        'The College of Integrated Chinese Medicine',
#        'The Edward James Foundation Limited',
#        'The Film Education Training Trust Limited',
#        'The Liverpool Institute for Performing Arts',
#        'The Royal Agricultural University',
#        'The Royal Central School of Speech and Drama',
#        'The University of Buckingham', 'The WKCIC Group', 'Tyne Coast College',
#        'University College of Osteopathy (The)', 'Warwickshire College'],
#       dtype='object', name='Provider name')
# Cluster 5:
# Index(['Aston University', 'Birkbeck College', 'Bournemouth University',
#        'Brighton and Sussex Medical School', 'Brunel University London',
#        'City, University of London', 'Edge Hill University',
#        'Glasgow Caledonian University', 'Heriot-Watt University',
#        'Imperial College of Science, Technology and Medicine',
#        'King's College London', 'LIBF Limited',
#        'Liverpool John Moores University', 'Loughborough University',
#        'Paris Dauphine International', 'Queen Mary University of London',
#        'Queen's University of Belfast',
#        'Royal Holloway and Bedford New College',
#        'St Mary's University College', 'Stranmillis University College',
#        'Swansea University',
#        'The London School of Economics and Political Science',
#        'The Royal Veterinary College', 'The University of Birmingham',
#        'The University of Bradford', 'The University of East Anglia',
#        'The University of Kent', 'The University of Lancaster',
#        'The University of Leeds', 'The University of Leicester',
#        'The University of Liverpool', 'The University of Manchester',
#        'The University of Warwick', 'University College London',
#        'University of Aberdeen', 'University of Brighton',
#        'University of Bristol', 'University of Cambridge',
#        'University of Dundee', 'University of Durham',
#        'University of Edinburgh', 'University of Exeter',
#        'University of Glasgow', 'University of Keele',
#        'University of Newcastle upon Tyne',
#        'University of Northumbria at Newcastle',
#        'University of Nottingham, The', 'University of Oxford',
#        'University of Plymouth', 'University of Southampton',
#        'University of Stirling', 'University of Strathclyde',
#        'University of Sussex', 'University of York',
#        'University of the West of England, Bristol'],
#       dtype='object', name='Provider name')
# Cluster
# 1    91.368077
# 2    84.988885
# 0    82.910801
# 3    79.873522
# 5    78.411416
# 4    74.005689
# dtype: float64