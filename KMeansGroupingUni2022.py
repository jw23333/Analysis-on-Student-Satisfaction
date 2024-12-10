import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
df_allSubjects_level1 = pd.read_excel('/Users/yiweiwang/Desktop/INF2-FDS/finalProject/NSS_registered_FT22_CAH.xlsx',header=3, 
sheet_name='NSSFULLTIME1', engine='openpyxl')
df_allSubjects_level2 = pd.read_excel('/Users/yiweiwang/Desktop/INF2-FDS/finalProject/NSS_registered_FT22_CAH.xlsx',header=3, 
sheet_name='NSSFULLTIME2', engine='openpyxl')
df_allSubjects_level3 = pd.read_excel('/Users/yiweiwang/Desktop/INF2-FDS/finalProject/NSS_registered_FT22_CAH.xlsx',header=3,
sheet_name='NSSFULLTIME3', engine='openpyxl')
df_allSubjects_concat = pd.concat([df_allSubjects_level1, df_allSubjects_level2, df_allSubjects_level3])
df_allSubjects_concat = df_allSubjects_concat.groupby(['Provider','Question Number'])['Actual value'].mean().reset_index()
drop_scale = ['Scale01','Scale02','Scale03','Scale04','Scale05','Scale06','Scale07','Scale08']
# Drop rows where the 'Question Number' column contains any value in drop_scale
df_allSubjects_concat = df_allSubjects_concat[~df_allSubjects_concat['Question Number'].isin(drop_scale)]
# Pivot the data so that each row is a university, each column is a question, and the values are the Actual value
df_allSubjects_pivot = df_allSubjects_concat.pivot(index='Provider', columns='Question Number', values='Actual value')
#check is there any NaN values
df_allSubjects_pivot = df_allSubjects_pivot.dropna(axis=0)
# Standardize the features
scaler = StandardScaler()
df_allSubjects_scaled = scaler.fit_transform(df_allSubjects_pivot)

# Apply PCA
pca = PCA()
df_allSubjects_pca = pca.fit_transform(df_allSubjects_scaled)

# Convert the PCA results into a DataFrame
df_allSubjects_pca = pd.DataFrame(df_allSubjects_pca, index=df_allSubjects_pivot.index)

# Calculate the cumulative sum of the explained variance ratio
cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

# Find the number of components that explain at least 95% of the variance
n_components_95 = np.where(cumulative_explained_variance >= 0.95)[0][0] + 1

# Get the top contributing features for each PC
top_features_all = set()
for i in range(n_components_95):
    top_features = np.abs(pca.components_[i]).argsort()[::-1][:5]
    top_features_all.update(df_allSubjects_pivot.columns[top_features])

# Select these columns in df_allSubjects_pivot
df_allSubjects_selected = df_allSubjects_pivot[list(top_features_all)].copy()
#now do the clustering
n_clusters = 6
np.random.seed(0)  # Set a random seed
kmeans = KMeans(n_clusters=n_clusters)
df_allSubjects_selected['Cluster'] = kmeans.fit_predict(df_allSubjects_selected)

# Print the provider names in each cluster
for i in range(n_clusters):
    print(f"Cluster {i}:")
    print(df_allSubjects_selected[df_allSubjects_selected['Cluster'] == i].index)

# Sort the clusters by the mean value of each feature
df_cluster_means = df_allSubjects_selected.groupby('Cluster').mean().mean(axis=1).sort_values(ascending=False)
print(df_cluster_means)
top_cluster = df_cluster_means.index[0]
top_cluster_content_2022 = df_allSubjects_selected[df_allSubjects_selected['Cluster'] == top_cluster].index.tolist()
print(top_cluster_content_2022)
#now do the bottom cluster
bottom_cluster = df_cluster_means.index[-1]
bottom_cluster_content_2022 = df_allSubjects_selected[df_allSubjects_selected['Cluster'] == bottom_cluster].index.tolist()
#results:
# Cluster 0:
# Index(['Abertay University',
#        'Anglia Ruskin University Higher Education Corporation',
#        'Architectural Association (Incorporated)',
#        'Arts University Bournemouth, the', 'Aston University',
#        'Bangor University', 'Bath Spa University', 'Birkbeck College',
#        'Birmingham City University', 'Bishop Burton College',
#        ...
#        'University of Wolverhampton', 'University of Worcester',
#        'University of York', 'University of the Arts, London',
#        'University of the West of England, Bristol',
#        'University of the West of Scotland', 'Wirral Metropolitan College',
#        'Writtle University College', 'York College',
#        'York St John University'],
#       dtype='object', name='Provider', length=140)
# Cluster 1:
# Index(['Activate Learning', 'Brockenhurst College',
#        'Central Bedfordshire College', 'City of Bristol College',
#        'City of Wolverhampton College', 'Moulton College',
#        'Mountview Academy of Theatre Arts Limited',
#        'Northeastern University - London', 'Nottingham College',
#        'Peter Symonds College', 'Plumpton College', 'Reaseheath College',
#        'South Gloucestershire and Stroud College', 'St Helens College',
#        'The Conservatoire for Dance and Drama', 'The Metanoia Institute',
#        'The SMB Group', 'Warwickshire College'],
#       dtype='object', name='Provider')
# Cluster 2:
# Index(['All Nations Christian College Limited', 'Amity Global Education Ltd',
#        'Barnet & Southgate College', 'Bishop Auckland College',
#        'Cardinal Newman College',
#        'Central School of Ballet Charitable Trust Limited(the)',
#        'Christ the Redeemer College',
#        'City and Guilds of London Art School Limited',
#        'City of Sunderland College', 'DCG', 'David Game College Ltd',
#        'EKC Group', 'Empire College London Limited',
#        'Futureworks Training Limited', 'Global Banking School Limited',
#        'Gloucestershire College', 'Grŵp Colegau NPTC Group of Colleges',
#        'Heart of Yorkshire Education Group', 'Hy Education Limited',
#        'ICON College of Technology and Management Ltd', 'Inter-ED UK Limited',
#        'LCCM AU UK Limited', 'London Bridge Business Academy Limited',
#        'London School of Commerce & IT Limited',
#        'London School of Management Education Limited',
#        'London School of Science & Technology Limited',
#        'Matrix College of Counselling and Psychotherapy Ltd',
#        'Middlesbrough College',
#        'Mont Rose College of Management and Sciences Limited',
#        'Moorlands College', 'Nazarene Theological College',
#        'Nelson College London Limited', 'New College Durham',
#        'New College Swindon', 'Northern College of Acupuncture',
#        'Northern Regional College', 'Northern School of Contemporary Dance',
#        'RTC Education Ltd', 'South Eastern Regional College',
#        'South West College', 'Southern Regional College', 'Spurgeon's College',
#        'The City of Liverpool College',
#        'The College of Integrated Chinese Medicine',
#        'The Markfield Institute of Higher Education',
#        'The Northern School of Art', 'The Oldham College',
#        'Truro and Penwith College', 'UCK Limited',
#        'Unified Seevic Palmer's College', 'United Colleges Group',
#        'University Centre Peterborough', 'West Herts College',
#        'Weymouth College', 'Wigan and Leigh College',
#        'Wiltshire College and University Centre'],
#       dtype='object', name='Provider')
# Cluster 3:
# Index(['ACM Guildford Limited', 'AECC University College', 'Lamda Limited',
#        'Lincoln College', 'Met Film School Limited', 'RNN Group',
#        'The Liverpool Institute for Performing Arts',
#        'The London Institute of Banking & Finance', 'The WKCIC Group'],
#       dtype='object', name='Provider')
# Cluster 4:
# Index(['Arden University Limited', 'BIMM University Limited',
#        'BPP University Limited', 'Backstage Academy (training) Ltd',
#        'City College Norwich', 'Colchester Institute', 'Cornwall College',
#        'Courtauld Institute of Art', 'Fareham College',
#        'Glasgow School of Art', 'Goldsmiths' College',
#        'Hull and York Medical School', 'London School of Theology',
#        'Pearson College Limited', 'Petroc', 'Preston College',
#        'Ravensbourne University London',
#        'Rose Bruford College of Theatre and Performance',
#        'Royal Academy of Dance', 'Royal Academy of Dramatic Art',
#        'Royal Conservatoire of Scotland',
#        'Royal Welsh College of Music and Drama', 'SAE Education Limited',
#        'School of Oriental and African Studies', 'Sparsholt College',
#        'St. George's Hospital Medical School',
#        'The Edward James Foundation Limited',
#        'The Film Education Training Trust Limited',
#        'The Royal Central School of Speech and Drama',
#        'Trinity Laban Conservatoire of Music and Dance',
#        'University College of Osteopathy (The)'],
#       dtype='object', name='Provider')
# Cluster 5:
# Index(['Aberystwyth University', 'Arts Educational Schools(The)',
#        'Arts University Plymouth', 'Askham Bryan College', 'Barnsley College',
#        'Belfast Metropolitan College', 'Bishop Grosseteste University',
#        'Blackburn College', 'Blackpool and the Fylde College',
#        'Bloomsbury Institute Limited', 'Boston College',
#        'Bournemouth and Poole College, The', 'Bradford College',
#        'Bridgwater and Taunton College', 'Buckinghamshire New University',
#        'Calderdale College', 'Cambridge Arts & Sciences Limited',
#        'Cheshire College South and West', 'Chichester College Group',
#        'City College Plymouth', 'Coventry University', 'DN Colleges Group',
#        'East Sussex College Group', 'Elim Foursquare Gospel Alliance',
#        'Exeter College', 'Farnborough College of Technology', 'ForMission Ltd',
#        'Grŵp Llandrillo Menai', 'Guildhall School of Music & Drama',
#        'Hartpury University', 'Heart of Worcestershire College',
#        'Hertford Regional College', 'Hugh Baird College', 'Hull College',
#        'ICMP Management Limited', 'LTE Group', 'Leeds Arts University',
#        'Leeds Conservatoire', 'Leeds Trinity University', 'Leicester College',
#        'London Metropolitan University', 'London South East Colleges',
#        'Loughborough College', 'Luminate Education Group',
#        'Morley College Limited', 'NCG', 'National Centre for Circus Arts',
#        'New City College', 'North East Surrey College of Technology (NESCOT)',
#        'North Warwickshire and South Leicestershire College',
#        'North West Regional College', 'Point Blank Limited',
#        'Rambert School of Ballet and Contemporary Dance', 'Riverside College',
#        'Sheffield College, The', 'Solihull College and University Centre',
#        'South Devon College',
#        'South Essex College of Further and Higher Education',
#        'South Thames Colleges Group', 'St Mary's University, Twickenham',
#        'St Mellitus College Trust', 'Stranmillis University College',
#        'TEC Partnership', 'Teesside University',
#        'The Cambridge Theological Federation',
#        'The Chicken Shed Theatre Trust', 'The University of Bolton',
#        'The University of Surrey', 'The University of West London',
#        'Trinity College (Bristol) Limited', 'University of Aberdeen',
#        'University of St Andrews', 'University of St Mark & St John',
#        'University of Ulster', 'University of the Highlands and Islands',
#        'Weston College of Further and Higher Education',
#        'Wrexham Glyndŵr University'],
#       dtype='object', name='Provider')
# Cluster
# 2    0.888148
# 5    0.805892
# 0    0.740575
# 1    0.687818
# 4    0.652475
# 3    0.559912
# dtype: float64