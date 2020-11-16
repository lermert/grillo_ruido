#import h5py
#import pandas
from sklearn.cluster import KMeans, MiniBatchKMeans

def cluster(df, n_clusters, n_iterations=100):

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(df)
    labels = kmeans.predict(df)
    centroids = kmeans.cluster_centers_
    return(labels, centroids, kmeans.inertia_, kmeans.n_iter_)

def cluster_minibatch(df, n_clusters, n_iterations):
    mbk = MiniBatchKMeans(init='k-means++',
                          n_clusters=n_clusters,
                          max_iter=n_iterations,
                          batch_size=int(len(df) / 10),
                          n_init=10,
                          max_no_improvement=20,
                          verbose=True,
                          random_state=0)
    mbk.fit(df)
    return(mbk.predict(df), mbk.cluster_centers_, mbk.inertia_, mbk.n_iter_)

# def cluster_agglo(

#   agglo = AgglomerativeClustering(n_clusters=c_clusters, affinity='precomputed',
#                                   linkage='average', distance_threshold=None))


# colors = list(map(lambda x: current_palette[x+1], labels))

# fig = plt.figure(figsize=(5, 15))
# for i in range(correlations_to_consider):
#     tsplot = np.zeros(nr_samples)
#     for j in range(nr_samples):
#         tsplot[j] = df.iat[i, j]
#     plt.plot(lag[ixs], tsplot + i * 1.e-19, color=colors[i], alpha=0.5)
# plt.xlabel("Lag (s)")
# plt.yticks([])
# plt.title("K-means clustering of traces from:\n{}\n{} clusters".format(os.path.basename(input_file), n_clusters))
# #plt.xlim(0, 30)
# plt.savefig("clusters.png")
# plt.show()
