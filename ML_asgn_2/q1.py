from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi

def calculate_clusters(X, Y_pred, centroids, k):
    """ Recalculates the clusters """
    # Initiate empty clusters
    clusters = {}
    # Set the range for value of k (number of centroids)
    for i in range(k):
        clusters[i] = []
    for data in X:
        euclid_dist = []
        for j in range(k):
            euclid_dist.append(np.linalg.norm(data - centroids[j]))
        # Append the cluster of data to the dictionary
        clusters[euclid_dist.index(min(euclid_dist))].append(data)
        centroid_dist = int((np.where(euclid_dist[:] == min(euclid_dist))[0]))
        Y_pred.append(centroid_dist)
        
    return clusters

def calculate_centroids(centroids, clusters, k):
    """ Recalculates the centroid position based on the plot """
    for i in range(k):
        centroids[i] = np.average(clusters[i], axis=0)
    return centroids
    
if __name__ == '__main__':

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])
    label_map = {'Iris-setosa' : 0,
                 'Iris-versicolor':1, 
                 'Iris-virginica':2}

    features = ['sepal length', 'sepal width', 'petal length', 'petal width']
    # Separating out the features
    x = df.loc[:, features].values
    # Separating out the target
    y = df.loc[:,['target']].values

    # Standardizing the features
    x = MinMaxScaler().fit_transform(x)


    pca = PCA()
    princ_comp = pca.fit_transform(x)
    x_pca = pca.transform(x)

    cov_mat = pca.get_covariance()
    print(cov_mat)
    relative_cov = pca.explained_variance_ratio_
    print(relative_cov)
    print(np.sum(relative_cov[:2])/np.sum(relative_cov))

    pca = PCA(n_components = 2)
    princ_comp = pca.fit_transform(x)
    princ_df = pd.DataFrame(data = princ_comp, columns = ['principal component 1', 'principal component 2'])
    x_pca = pca.transform(x)

    final_df = pd.concat([princ_df, df[['target']]], axis = 1)

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    colors = ['r', 'g', 'b']
    for target, color in zip(targets,colors):
        indicesToKeep = final_df['target'] == target
        ax.scatter(final_df.loc[indicesToKeep, 'principal component 1'], final_df.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)
        ax.legend(targets)
        ax.grid()

    plt.show()
    df['target'].replace(label_map,inplace = True)
    Y = df['target']
    NMI = []

    for k in range(2, 9):
        clusters = {}
        for i in range(k):
            clusters[i] = []
        centroids = {}
        for i in range(k):
            centroids[i] = x_pca[i]
        for data in x_pca:
            euclid_dist = []
            for j in range(k):
                euclid_dist.append(np.linalg.norm(data - centroids[j]))
            clusters[euclid_dist.index(min(euclid_dist))].append(data)
        goahead = True
        while(goahead):
            new_centroids = calculate_centroids(x_pca, clusters, k)
            if np.array_equal(centroids, new_centroids):
                goahead = False 
            Y_pred = []
            clusters = calculate_clusters(x_pca, Y_pred, new_centroids, k)
            centroids = new_centroids
        NMI.append(nmi(Y, Y_pred))

    K = np.arange(2,9,dtype = int)

    plt.plot(K, NMI)
    plt.xlabel('K')
    plt.ylabel('Normalized mutual information')
    plt.title('k vs NMI')
    plt.show()



