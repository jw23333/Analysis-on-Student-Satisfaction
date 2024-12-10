import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
df_allSubjects_level1 = pd.read_excel('/Users/yiweiwang/Desktop/INF2-FDS/finalProject/NSS_registered_FT21_CAH.xlsx',header=3, 
sheet_name='NSSFULLTIME1', engine='openpyxl')
df_allSubjects_level2 = pd.read_excel('/Users/yiweiwang/Desktop/INF2-FDS/finalProject/NSS_registered_FT21_CAH.xlsx',header=3, 
sheet_name='NSSFULLTIME2', engine='openpyxl')
df_allSubjects_level3 = pd.read_excel('/Users/yiweiwang/Desktop/INF2-FDS/finalProject/NSS_registered_FT21_CAH.xlsx',header=3,
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
#store the top cluster as a list
top_cluster = df_cluster_means.index[0]
top_cluster_content_2021 = df_allSubjects_selected[df_allSubjects_selected['Cluster'] == top_cluster].index.tolist()
print(df_cluster_means)
#now do the bottom cluster
bottom_cluster = df_cluster_means.index[-1]
bottom_cluster_content_2021 = df_allSubjects_selected[df_allSubjects_selected['Cluster'] == bottom_cluster].index.tolist()
#results:
# yiweiwang@YiweideMacBook-Pro finalProject % /usr/local/bin/python3 /Users/yiweiwang/Desktop/INF2-FDS/finalProject/KMeansGrou
# pingUni2021.py
# Cluster 0:
# Index(['Abertay University',
#        'Anglia Ruskin University Higher Education Corporation',
#        'Aston University', 'BPP University Limited', 'Bangor University',
#        'Bath Spa University', 'Belfast Metropolitan College',
#        'Birmingham City University', 'Bradford College',
#        'Bridgwater and Taunton College',
#        ...
#        'University of Strathclyde', 'University of Sussex',
#        'University of Ulster', 'University of Wolverhampton',
#        'University of York', 'University of the West of England, Bristol',
#        'University of the West of Scotland', 'West Herts College',
#        'Wirral Metropolitan College', 'York St John University'],
#       dtype='object', name='Provider', length=122)
# Cluster 1:
# Index(['Amity Global Education Ltd', 'Arts Educational Schools(the)',
#        'Brighton and Sussex Medical School', 'Brockenhurst College',
#        'Cardinal Newman College', 'City College Plymouth',
#        'City and Guilds of London Art School Limited',
#        'City of Sunderland College', 'Cliff College',
#        'Empire College London Limited', 'ForMission Ltd',
#        'Global Banking School Limited', 'Hy Education Limited',
#        'Kensington Education Foundation Limited',
#        'London Bridge Business Academy Limited',
#        'London Churchill College Ltd', 'London College of Business Studies',
#        'London School of Commerce & IT Limited',
#        'London School of Management Education Limited', 'Moorlands College',
#        'Nazarene Theological College', 'Nelson College London Limited',
#        'The City of Liverpool College', 'The Kingham Hill Trust',
#        'The Markfield Institute of Higher Education',
#        'Trinity College (Bristol) Limited', 'UCK Limited',
#        'United Colleges Group', 'York College'],
#       dtype='object', name='Provider')
# Cluster 2:
# Index(['Barnsley College', 'Bloomsbury Institute Limited',
#        'Calderdale College', 'Cambridge Arts & Sciences Limited',
#        'Cheshire College South and West', 'Dudley College of Technology',
#        'Fairfield School of Business Ltd', 'Fareham College',
#        'Futureworks Training Limited', 'Heart of Yorkshire Education Group',
#        'Hopwood Hall College', 'Inter-ED UK Limited', 'Le Cordon Bleu Limited',
#        'Lincoln College', 'London School of Science & Technology Limited',
#        'Luminate Education Group',
#        'Matrix College of Counselling and Psychotherapy Ltd',
#        'Middlesbrough College',
#        'Mont Rose College of Management and Sciences Limited',
#        'Morley College Limited', 'New College Durham',
#        'Norland College Limited',
#        'North East Surrey College of Technology (NESCOT)',
#        'Northern College of Acupuncture', 'RNN Group', 'RTC Education Ltd',
#        'Royal Academy of Dance', 'Sheffield College, The',
#        'South Gloucestershire and Stroud College', 'St Helens College',
#        'St Mellitus College Trust', 'The Cambridge Theological Federation',
#        'The Chicken Shed Theatre Trust',
#        'The College of Integrated Chinese Medicine', 'The Oldham College',
#        'The SMB Group', 'The Trafford College Group',
#        'University Centre Quayside Limited', 'University of St Andrews',
#        'Weston College of Further and Higher Education',
#        'Wigan and Leigh College', 'Writtle University College'],
#       dtype='object', name='Provider')
# Cluster 3:
# Index(['ALRA', 'Arden University Limited', 'Askham Bryan College',
#        'BIMM Limited', 'Bedford College',
#        'Berkshire College of Agriculture, the (BCA)', 'Birkbeck College',
#        'Bishop Burton College', 'Bishop Grosseteste University',
#        'Bournemouth University', 'Brunel University London', 'Bury College',
#        'Chichester College Group', 'Colchester Institute', 'Cornwall College',
#        'Court Theatre Training Company Ltd', 'Courtauld Institute of Art',
#        'De Montfort University', 'EKC Group', 'Gateshead College',
#        'Glasgow School of Art', 'Goldsmiths' College', 'Hull College',
#        'ICMP Management Limited', 'Kingston Maurward College', 'LTE Group',
#        'Leeds Conservatoire', 'Leeds Trinity University',
#        'Liverpool Hope University', 'London School of Theology',
#        'London South Bank University', 'Manchester Metropolitan University',
#        'Met Film School Limited', 'Mountview Academy of Theatre Arts Limited',
#        'New City College',
#        'North Warwickshire and South Leicestershire College',
#        'Plumpton College', 'Ravensbourne University London',
#        'Reaseheath College', 'Rose Bruford College of Theatre and Performance',
#        'SAE Education Limited',
#        'South Essex College of Further and Higher Education',
#        'Southampton City College', 'Southern Regional College',
#        'St. George's Hospital Medical School',
#        'The Film Education Training Trust Limited',
#        'The London Institute of Banking & Finance', 'The Metanoia Institute',
#        'The Royal Agricultural University',
#        'The Royal Central School of Speech and Drama',
#        'The Royal Veterinary College', 'The University of Birmingham',
#        'The University of Manchester',
#        'Trinity Laban Conservatoire of Music and Dance', 'Tyne Coast College',
#        'University of Brighton', 'University of Edinburgh',
#        'University of Newcastle upon Tyne', 'University of Northampton, The',
#        'University of Northumbria at Newcastle', 'University of Suffolk',
#        'University of Winchester', 'University of the Arts, London',
#        'Warwickshire College'],
#       dtype='object', name='Provider')
# Cluster 4:
# Index(['ACM Guildford Limited', 'AECC University College', 'Activate Learning',
#        'Bournemouth and Poole College, The', 'Lamda Limited',
#        'Nottingham College', 'Pearson College Limited', 'Preston College',
#        'South Thames Colleges Group', 'The WKCIC Group'],
#       dtype='object', name='Provider')
# Cluster 5:
# Index(['Aberystwyth University', 'Architectural Association (Incorporated)',
#        'Arts University Bournemouth, the', 'Barnet & Southgate College',
#        'Blackburn College', 'Blackpool and the Fylde College',
#        'Boston College', 'Central Bedfordshire College',
#        'City of Bristol College', 'Coventry College', 'Coventry University',
#        'Craven College', 'Croydon College', 'DCG', 'East Sussex College Group',
#        'Elim Foursquare Gospel Alliance', 'Exeter College', 'Furness College',
#        'Greater Brighton Metropolitan College', 'Grŵp Llandrillo Menai',
#        'Guildhall School of Music & Drama', 'Hartpury University',
#        'Heart of Worcestershire College', 'Hereford College of Arts',
#        'Holy Cross College', 'ICON College of Technology and Management Ltd',
#        'LCCM AU UK Limited', 'Leicester College',
#        'London Metropolitan University', 'London South East Colleges',
#        'London Studio Centre Limited', 'Loughborough College',
#        'Loughborough University', 'Moulton College', 'Myerscough College',
#        'NCG', 'NCH at Northeastern Limited', 'New College Swindon',
#        'Newman University', 'Petroc', 'Point Blank Limited',
#        'Queen Margaret University, Edinburgh', 'Royal Academy of Dramatic Art',
#        'Royal College of Music', 'SRUC',
#        'Solihull College and University Centre',
#        'South & City College Birmingham', 'South Devon College',
#        'South Eastern Regional College', 'Southport College',
#        'St Mary's University, Twickenham',
#        'St. Patrick's International College Limited',
#        'Staffordshire University', 'TEC Partnership',
#        'The Conservatoire for Dance and Drama',
#        'The Liverpool Institute for Performing Arts', 'The Salvation Army',
#        'The University of Bolton', 'The University of Chichester',
#        'The University of West London', 'Truro and Penwith College',
#        'University for the Creative Arts', 'University of St Mark & St John',
#        'University of Sunderland', 'University of Wales Trinity Saint David',
#        'University of Worcester', 'University of the Highlands and Islands',
#        'Wrexham Glyndŵr University', 'Yeovil College'],
#       dtype='object', name='Provider')
# Cluster
# 1    0.923779
# 2    0.840359
# 5    0.769855
# 0    0.728937
# 3    0.671368
# 4    0.542890
# dtype: float64