import numpy as np 
import os

try:
    import cPickle as _pickle
except ImportError:
    import pickle as _pickle


from sklearn.cluster import DBSCAN, 
from sklearn.cluster._dbscan_inner import dbscan_inner
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
from gensim.models import Word2Vec


class MyDBSCAN(DBSCAN):
    def __init__(self, eps=0.5, *, min_samples=10, w2v_model_path):
        super().__init__(eps=eps, min_samples=min_samples)
        self.w2v_model = Word2Vec.load(w2v_model_path)

    # 对原来的fit进行一点，解析并保留X数据
    def fit(self, X, y=None, sample_weight=None):
        X = list(filter(lambda x:self.w2v_model.wv.__contains__(x[0]), X))
        self.tags, self.reviews_num = zip(*X)
        self.X = np.array([self.w2v_model.wv.__getitem__(t) for t in self.tags])
        X = self.X

        # X = self._validate_data(X, accept_sparse='csr')

        if not self.eps > 0.0:
            raise ValueError("eps must be positive.")

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        # Calculate neighborhood for all samples. This leaves the original
        # point in, which needs to be considered later (i.e. point i is in the
        # neighborhood of point i. While True, its useless information)
        if self.metric == 'precomputed' and sparse.issparse(X):
            # set the diagonal to explicit values, as a point is its own
            # neighbor
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', sparse.SparseEfficiencyWarning)
                X.setdiag(X.diagonal())  # XXX: modifies X's internals in-place

        neighbors_model = NearestNeighbors(
            radius=self.eps, algorithm=self.algorithm,
            leaf_size=self.leaf_size, metric=self.metric,
            metric_params=self.metric_params, p=self.p, n_jobs=self.n_jobs)
        neighbors_model.fit(X)
        # This has worst case O(n^2) memory complexity
        neighborhoods = neighbors_model.radius_neighbors(X,
                                                         return_distance=False)

        if sample_weight is None:
            n_neighbors = np.array([len(neighbors)
                                    for neighbors in neighborhoods])
        else:
            n_neighbors = np.array([np.sum(sample_weight[neighbors])
                                    for neighbors in neighborhoods])

        # Initially, all samples are noise.
        labels = np.full(X.shape[0], -1, dtype=np.intp)

        # A list of all core samples found.
        core_samples = np.asarray(n_neighbors >= self.min_samples,
                                  dtype=np.uint8)
        dbscan_inner(core_samples, neighborhoods, labels)

        self.core_sample_indices_ = np.where(core_samples)[0]
        self.labels_ = labels

        if len(self.core_sample_indices_):
            # fix for scipy sparse indexing issue
            self.components_ = X[self.core_sample_indices_].copy()
        else:
            # no core samples
            self.components_ = np.empty((0, X.shape[1]))
        return self

    def get_cluster_info(self, label, show_tags_num=5):
        label_indices = np.where(self.labels_ == label)[0]
        cluster_X = self.X[label_indices]
        cluster_size = len(label_indices)

        # ========== centers信息 ==========
        fake_center = np.mean(cluster_X, axis=0)
        real_center_index = np.argmin([np.linalg.norm(x-fake_center) for x in cluster_X])
        center_x = cluster_X[real_center_index]

        center_to_others = sorted(enumerate([np.linalg.norm(x-center_x) for x in cluster_X]), 
                                    key=lambda p:p[1], reverse=True)
        # 包括center自己
        center_neighbors_index = np.array([p[0] for p in center_to_others])
        centers_tags = list(set([self.tags[x] for x in label_indices[center_neighbors_index]]))
        centers_tags = centers_tags[:show_tags_num]

        # ========== reviews_num ==========
        cluster_avg_reviews_num = np.mean([self.reviews_num[x] for x in label_indices])

        return label, cluster_size, cluster_avg_reviews_num, centers_tags


    def report_clustering_result(self):
        size = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        print("{} clusters.".format(size))
        clusters_info = []
        for i in range(size):
            clusters_info.append(self.get_cluster_info(i))
        clusters_info = sorted(clusters_info, key=lambda p:p[1], reverse=True)

        single_report = """
=====================================
cluster[{}] - size: {}
=====================================
avg_reviews_num: {:.2f},
centers: {}

        """

        for info in clusters_info[:50]:
            print(single_report.format(*info))

    def save(self, save_path, pickle_protocol=2):
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        with open(save_path,'wb') as f:
            try:
                _pickle.dump(self,f,protocol=pickle_protocol)
                print("successfully saved clusters set.")
            except Exception as e:
                print("failed to save clusters set.")
                print("ERROR:",e)

    @classmethod
    def load(cls, load_path):
        with open(load_path,'rb') as f:
            print("loading {} object from {}".format(cls.__name__,load_path))
            obj = _pickle.load(f)
            print("successfully loaded.")
        return obj


class MyKMeans()


def test():
    w2v_model_path = '../models/word2vec/abs_word2vec_2.5.mod'
    cluster_model_path = '../models/DBSCAN_clusters/DBSCAN_clusters_model.pkl'
    data_path = '../data/tags_pool/breakouts_tags_100gg7_t10.pkl'
    with open(data_path,'rb') as f:
        data = _pickle.load(f)

    mdb = MyDBSCAN(eps=6,min_samples=20,w2v_model_path=w2v_model_path)
    mdb.fit(data)
    mdb.save(save_path=cluster_model_path)

    mdb = MyDBSCAN.load(cluster_model_path)
    mdb.report_clustering_result()


if __name__ == '__main__':
    test()
