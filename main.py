from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
from sklearn.cluster import KMeans


def read_data(filename):
    return pd.read_csv(filename)


if __name__ == "__main__":
    df = read_data("alls.csv")
    content = df["Content"].str.split().tolist()
    print(df.head())
    # print(content[:2])
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(content)]
    # print(documents[:2])
    # save to file
    import os
    model = None
    if os.path.exists("my_doc2vec.d2v"):
        model = Doc2Vec.load("my_doc2vec.d2v")
    else:
        model = Doc2Vec(documents, vector_size=100,
                        window=2, min_count=1, workers=4)
    # from gensim.test.utils import get_tmpfile
        model.save("my_doc2vec.d2v")
    # check
    # vector = model.infer_vector(documents[1][0])

    # change here
    k = 5

    from sklearn.cluster import Birch
    # inference hyper-parameters
    start_alpha = 0.01
    infer_epoch = 1000
    # get all documents vector
    X = [model.infer_vector(d[0], alpha=start_alpha,
                            steps=infer_epoch) for d in documents]

    brc = Birch(branching_factor=50, n_clusters=k,
                threshold=0.1, compute_labels=True)
    brc.fit(X)

    clusters = brc.predict(X)

    labels = brc.labels_

    print("Clusters: ")
    print(clusters)

    # silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')

    # print("Silhouette_score: ")
    # print(silhouette_score)

    clustered_docs = zip(clusters, [" ".join(d) for d in content])
    df_c = pd.DataFrame(clustered_docs, columns=["cluster", "docs"])
    df_c.to_csv("clustered_docs.csv")
    print("data saved")
