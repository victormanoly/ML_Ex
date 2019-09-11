import pandas as pd
import numpy as np
import operator
import warnings
warnings.filterwarnings("ignore")

def dist (a, b):
  aux = a - b
  dist_2 =  0
  dist_2 = np.dot(aux.T,aux)
  return np.sqrt(dist_2)

# Count neighbors  
def get_class(neighbors):
  class_array = {}
  for i in range(len(neighbors)):
    rotulo = neighbors[i][-1]
    if rotulo in class_array:
      class_array[rotulo] += 1
    else:
	    class_array[rotulo] = 1
  sorted_class = sorted(class_array.items(), key=operator.itemgetter(1), reverse=True)
  return sorted_class[0][0]
  
# KNN
def knn(data_train, sample, k):
  dist_array = {}
  train_size = len(data_train)
  for i in range(train_size):
    dist_i_sample = dist(data_train[i,:-1], sample[:-1])
    dist_array[i] = dist_i_sample
  neighbors = sorted(dist_array, key=dist_array.get)[:k]
  return get_class(data_train[neighbors])

def treat(data):
  idx = np.where(data == '?') 
  for i in range(len(idx[0])):
    sum0, aux = 0.0, 0.0
    for j in range(len(data)):
      if data[j][idx[1][i]] != '?':
        data[j][idx[1][i]] = float(data[j][idx[1][i]])
        sum0 += data[j][idx[1][i]]
        aux += 1
      mean = sum0 / aux
      if idx[1][i] != len(data[0]) - 1:
        data[idx[0][i]][idx[1][i]] = mean
      else:
        data[idx[0][i]][idx[1][i]] = int(mean)
  return data

###  Ler dados treino
df0 = pd.read_csv(
    filepath_or_buffer='nebulosa_train.txt',
    header=None,
    sep=' ')
df0.dropna(how="all", inplace=True)
df0.tail()
data_train = df0.loc[:,2:].values

# Ler dados teste
df1 = pd.read_csv(
    filepath_or_buffer='nebulosa_test.txt',
    header=None,
    sep=' ')
df1.dropna(how="all", inplace=True)
df1.tail()
data_test = df1.loc[:,2:].values

# Substituir as '?'
data_train = treat(data_train)
data_test  = treat(data_test)

# NN
N = np.shape(data_test)[0]
correct_class   = 0
for sample in data_test:
    knn_class = knn(data_train, sample, 1)
    if sample[-1] == knn_class:
        correct_class += 1

print('\n-------- NN --------')
print('Acurácia: {:0.2f}'.format((100 * correct_class / N)))
print('--------------------\n')
 
###### Rocchio
# Train Rocchio
def train_rocchio(train_samples):
  aux1, aux2, aux3 = 0, 0, 0
  sum1 = np.zeros(np.shape(train_samples[0,:-1]))
  sum2 = np.zeros(np.shape(train_samples[0,:-1]))
  sum3 = np.zeros(np.shape(train_samples[0,:-1]))
  for sample in train_samples:
    if sample[-1] == 1:
      aux1 += 1
      sum1 = sum1 + sample[:-1]
    elif sample[-1] == 2:
      aux2 += 1
      sum2 = sum2 + sample[:-1]
    elif sample[-1] == 3:
      aux3 += 1
      sum3 = sum3 + sample[:-1]
    else:
      return False
  centroid1 = sum1 / aux1
  centroid2 = sum2 / aux2
  centroid3 = sum3 / aux3
  return centroid1, centroid2, centroid3

def test_rocchio(data,centroid1,centroid2,centroid3):
  N = np.shape(data)[0]
  correct_class = 0
  for i in range(N):
    dist1 = dist(data[i,:-1],centroid1)  
    dist2 = dist(data[i,:-1],centroid2)
    dist3 = dist(data[i,:-1],centroid3)
    closest = 1 + np.argmin([dist1,dist2,dist3])
    if closest == 1:
      if data[i,-1] == 1:
        correct_class += 1
    elif closest == 2:
      if data[i,-1] == 2:
        correct_class += 1
    elif closest == 3:
      if data[i,-1] == 3:
        correct_class += 1
    else:
        return False      
  accuracy  = (correct_class / N) * 100
  return accuracy

# Train Rocchio
centroid1, centroid2, centroid3 = train_rocchio(data_train)
# Test
accuracy = test_rocchio(data_test,centroid1,centroid2,centroid3)
print('----- Rocchio ------')
print('Acurácia: {:0.2f}'.format(accuracy))
print('--------------------\n')

print('------ Comentário ------')
print("O tratamento proposto é buscar os dados\n",
"que estão faltando (?) e substituir pela\n",
      "média de valores da coluna.")