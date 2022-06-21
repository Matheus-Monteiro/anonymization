import time
from datetime import timedelta
import numpy as np
import pandas as pd
import sklearn.decomposition

start_time = time.time()

dataset = pd.read_csv('df_original_100000.csv')

label = dataset['Label']
del dataset['Label']
mu = np.mean(dataset, axis=0)
column_names = dataset.columns

print('############# ORIGINAL ###################')
print(dataset.head(5))

print('\n')

pca = sklearn.decomposition.PCA()
pca.fit(dataset)

number_of_components = 79

principalComponents = pca.transform(dataset)[:,:number_of_components]
components = pca.components_[:number_of_components,:]
Z = []


for v in components:
    my_mean, my_std = np.mean(v), np.std(v)
    x = np.random.normal(loc=my_mean, scale=my_std, size=79)
    Z.append(x)


generated_dataset = np.dot(principalComponents, Z)
generated_dataset += mu

print('############# ANONYMIZED ###################')
anonymizationDf = pd.DataFrame(data=generated_dataset, columns=column_names)

anonymizationDf.insert(loc=79,column='Label', value=label)

print(anonymizationDf.head(5))

elapsed_time_secs = time.time() - start_time
print("Execution took: %s secs (Wall clock time)" % timedelta(seconds=round(elapsed_time_secs)))