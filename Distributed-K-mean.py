
from mpi4py import MPI
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import time


T1 = time.time()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.size

#read data partially because of memory limitation
cat = ['alt.atheism','comp.graphics'] #, 'comp.os.ms-windows.misc']

newsgroups = fetch_20newsgroups(categories=cat)
#vectorize the text, convert the texts to vectors
vectorizer = TfidfVectorizer(stop_words='english')
data = vectorizer.fit_transform(newsgroups.data)
data = data.toarray()
print(data)
np.random.seed(10)
#data = np.random.randint(10,size=(100,5))

def newcentroid(data_class):
    newcentroid=[]
    for i in data_class: 
        c=np.mean(data_class[i], axis=0)
        newcentroid.append(c)
    return newcentroid

k = 4
max_itr=5

centroids= []

for i in range(k):
    centroids.append(data[i])

print("All centroids: ",centroids)

for i in range(max_itr):
    print("Iteration # ",i)
    centroids = comm.bcast(centroids, root=0)
    if rank == 0:
    #   print("whole data:",data)
        sendbuf = np.array_split(data,size,axis=0)
        data_class = None
    else:
        sendbuf= None

    recbuf=comm.scatter(sendbuf, root=0) 
    
    #    print(centroids)
    #    print(recbuf)
    #classification will be a dictionary of list
    classifications = {}
    print("centroids for me",rank,(centroids))         
    for i in range(k):
        classifications[i] = []
    for featureset in recbuf:
        #Euclidean Distance
        distances= [ np.linalg.norm(featureset-centroids[centroid]) for centroid in range(k)]
        #index of minimum distances 
        classification = distances.index(min(distances))
        #update the membership
        classifications[classification].append(featureset)
    data_class = comm.gather(classifications,root=0)
#    print("class data",(data_class))
        
    if rank ==0:
        
        classes = {}
        #gather classification 
        for dc in data_class:
            for c in dc:
                if c not in classes:
                    #no class has found for the object
                    classes[c] = []
                classes[c] = classes[c] + dc[c]
        newcentroids = newcentroid(classes)
        centroids = newcentroids
        print("New Centeroid:",centroids)
        print("Time Consumption: ", time.time()-T1)
