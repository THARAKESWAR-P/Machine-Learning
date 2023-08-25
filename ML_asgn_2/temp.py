from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from numpy import genfromtxt
import matplotlib.pyplot as plot
import pandas as pd
import numpy

def EuclideanDistance(p1,p2):
    dist=0.0
    for i in range(2):
        dist = dist + (float(p1[i])-float(p2[i]))**2
    return numpy.sqrt(dist)

def calc_jaccard_distance(bucket,index_list,unique_list):
    
    for i in range(3):
        dist = 1.0
        intersection = numpy.intersect1d(bucket,index_list[i]).shape[0]
        union = numpy.union1d(bucket,index_list[i]).shape[0]
        dist -= intersection/union
        print("\tJaccard dist with ",  unique_list[i], " is : ", dist)

def print_jaccard_distance(bucket, mean, index_list, unique_list):
    for i in range(3):
        print("\ncluster : ",i,"\n\tMean : ", mean[i],"\n")
        calc_jaccard_distance(bucket[i],index_list,unique_list)
        
def get_index_list(n):
    index_list = []
    for i in range(n):
        index_list.append(i)
    return index_list

def get_cluster(n, mean, TE_X):
    cluster=[[],[],[]]
            
    for i in range(n):
        dist=0.000000
        select=0
        min=10000000.000
        for j in range(3):
            dist=EuclideanDistance(mean[j],TE_X[i])
            if(dist<min):
                select = j
                min = dist
        cluster[select].append(i)
    return cluster

def get_new_mean(cluster, TE):
    mean=[]
    m = []
    for i in range(3):
        m = [float(0.0), float(0.0)]
        for ind in cluster[i]:
            #print(row)
            for k in range(2):
                m[k]=m[k]+float(TE[ind][k])

        for k in range(2):
            m[k]=m[k]/len(cluster[i])

        mean.append(m)
    return mean
    

if __name__ == '__main__':

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

    df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])

    features = ['sepal length', 'sepal width', 'petal length', 'petal width']
    # Separating out the features
    x = df.loc[:, features].values
    # Separating out the target
    y = df.loc[:,['target']].values

    #print(x)

    # Standardizing the features
    #x = StandardScaler().fit_transform(x)
    #print(x)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
    x_pca = pca.transform(x)

    finalDf = pd.concat([principalDf, df[['target']]], axis = 1)

    fig = plot.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    colors = ['r', 'g', 'b']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'], finalDf.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)
        ax.legend(targets)
        ax.grid()

    pca.explained_variance_ratio_

    print(x)
    print(x_pca)

    #TE = genfromtxt('iris.data', delimiter=',', dtype = str)
    TE = x_pca

    n = TE.shape[0]
    print("Dataset Size - ",n)

    index_list=get_index_list(n)

    randomPointIndex = numpy.random.choice(index_list,3)
    
    randomPoints=[]
    for i in range(3):
        randomPoints.append(list(TE[randomPointIndex[i]]))
    print("Random Points taken - ",randomPoints)

    mean=randomPoints
    for itr in range(10):
        cluster = get_cluster(n, mean, TE)
        mean = get_new_mean(cluster,TE)


    unique_clusters=numpy.unique(y)
    orig_cluster_index_list=[[],[],[]]
    
    for i in range(n):
        index = list(numpy.where(unique_clusters==y[i]))[0][0]
        orig_cluster_index_list[index].append(i)

    unique_clusters=list(unique_clusters)
    print_jaccard_distance(cluster, mean, orig_cluster_index_list, unique_clusters)

    #for k in range(2, 9):
        

    plot.show()



