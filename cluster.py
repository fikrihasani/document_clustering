import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN
from collections import Counter

def clusterBirch(content, data):
    k = 5

    brc = Birch(branching_factor=50, n_clusters=k,
                threshold=0.1, compute_labels=True)
    brc.fit(data)

    clusters = brc.predict(data)

    # labels = brc.labels_

    saveData("BIRCH", clusters, content)

def clusterDbscan(content, data):
    dbs = DBSCAN(eps=1.5, min_samples=2)
    dbs.fit(data)

    clusters = dbs.labels_

    saveData("DBSCAN", clusters, content)

def saveData(name, clusters, content):
    print(name, "Clusters: ")
    print(clusters)

    print(name, "Clusters: ")
    print(Counter(clusters).keys()) # equals to list(set(words))

    print(name, "Clusters freq: ")
    print(Counter(clusters).values()) # counts the elements' frequency
    
    clustered_docs = zip(clusters, [" ".join(d) for d in content])
    df_c = pd.DataFrame(clustered_docs, columns=["cluster", "docs"])
    df_c.to_csv(name+"_clustered_docs.csv")
    print(name, "data saved")
