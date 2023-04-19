import numpy as np
from sklearn import mixture
from sklearn import cluster


class Detector:
    def __init__(self):
        self.defense = None

    @property
    def scotts_factor(self):
        try:
            return self._scotts_factor
        except AttributeError:
            n, d = self.X.shape
            self._scotts_factor = np.power(n, -1. / (d + 4))
            return self._scotts_factor

    @property
    def threshold(self):
        try:
            return self._threshold
        except AttributeError:
            self._threshold = self.pdf(self.X).min()
            return self._threshold

    def fit(self, X):
        self.X = X
        self.tree = self.defense.tree

    def predict(self, transitions):
        return self._predict(self.defense.process_transitions(transitions))

    def _predict(self, transitions):
        """
        Compares the result from predictProba, if it does not exceeds threshold, is considered an attack

        Parameters
        ----------
        transitions : np.array
            Train transitions

        Returns
        -------
        bool
            0 (non-attack) if predictProba exceeds threshold else 1 (attack)
        """
        return self._predictProba(transitions) < 0.5

    def predictProba(self, transitions):
        return self._predictProba(self.defense.process_transitions(transitions))

    def _predictProba(self, transitions):
        """
        Based on a probability density function, estimates the probability for a transition of being normal

        Parameters
        ----------
        transitions : np.array
            Train transitions

        Returns
        -------
        float
            if threshold is None then probability_density_function
            if threshold <= 0 then 1, (100% normal transition)
            if threshold >= inf then 0 (100% attacked transitions)
            if threshold > 0 then probability_density_function / (2 * threshold)
        """
        pdf = self.pdf(transitions)

        if self.threshold is not None:
            if 0 < self.threshold < np.inf:
                return np.clip(pdf / (2 * self.threshold), 0, 1)
            elif self.threshold == np.inf:
                return np.zeros(len(transitions))
            else:
                return np.ones(len(transitions))
        else:
            return pdf

    def pdf(self, transitions):
        raise NotImplementedError
    
    @staticmethod
    def dist(x, y):
        return np.sqrt(((x[:, None] - y) ** 2).sum(axis=-1))


class DBSCAN(Detector):
    def fit(self, X):
        super().fit(X)
        self.epsilon = self.scotts_factor

    def pdf(self, transitions):
        neighbors_inside_range = self.tree.query_radius(transitions, self.epsilon, count_only=True)
        return neighbors_inside_range


class GaussianMixture(Detector):
    """
    Gaussian Mixture is a probabilistic model in which observations are considered to follow a probabilistic
    function formed by the combination of several formal distributions. It approximates to a kmeans generalization in
    which, instead of assigning each observation to a single cluster, a probability distribution of belonging
    to each one is obtained.

    ...

    Attributes
    ----------
    n_clusters : int
        Number of clusters used to study data
    model : GaussianMixture instance
        Model built over training transitions

    Methods
    -------
    fit(X)
        Initializes GaussianMixture model
    predict(transitions)
        The higher the density value of an obs, the more evidence that the obs belongs to a certain distribution
        Often, to facilitate calculations, instead of using the density value, the logarithm is used
    """
    def __init__(self, n_clusters=1024):
        super().__init__()
        self.n_clusters = n_clusters

    def fit(self, X):
        """
        Initializes GaussianMixture model

        Parameters
        ----------
        X : np.array
            Train transitions
        """
        super().fit(X)
        self.model = mixture.GaussianMixture(n_components=self.n_clusters).fit(X)

    def pdf(self, transitions):
        """
        The higher the density value of an obs, the more evidence that the obs belongs to a certain distribution
        Often, to facilitate calculations, instead of using the density value, the logarithm is used

        Parameters
        ----------
        transitions : np.array
            Train transitions

        Returns
        -------
        float
            e^x, x = Gaussian Mixture density value of the transition
        """
        return np.exp(self.model.score_samples(transitions))


class KernelDensity(Detector):
    """
    Kernel density estimation (KDE) is a non-parametric way to estimate the probability density function of a random
    variable. The best way to understand how it works is checking out an example:
    Text example: https://en.wikipedia.org/wiki/Kernel_density_estimation#Example
    Visual example: https://www.youtube.com/watch?v=DCgPRaIDYXA&ab_channel=KimberlyFessel

    ...

    Attributes
    ----------
    h : float
        Kernel's bandwidth, the higher it is, the smoother the probability density function becomes

    Methods
    -------
    pdf(transitions)
        Calculates the probability density function for a transaction given by KDE
    """

    def fit(self, X):
        super().fit(X)
        self.h = self.scotts_factor

    def pdf(self, transitions):
        """
        Calculates the probability density function for a transaction given by KDE

        Parameters
        ----------
        transitions : np.array
            Train transitions

        Returns
        -------
        float
            Probability density function
        """
        return self.tree.kernel_density(transitions, self.h)


class KMeans(Detector):
    """
    K-Means predictor implementation (probability predictor)

    ...

    Attributes
    ----------
    n_clusters : int
        Number of clusters used to group data

    Methods
    -------
    fit(X)
        Groups data into n_clusters using k-means algorithm
        Saves centroids to a BallTree

    pdf(transitions)
        For a given transition, distance to nearest centroid is queried
        A probability density function is calculated as follows (1 / (distance_to_nearest_centroid + 1))
    """
    def __init__(self, n_clusters=1024, beta=1, use_beta=False):
        super().__init__()
        self.n_clusters = n_clusters
        self.centroids = None
        self.centroid_tree = None
        self.beta = beta
        self.use_beta = use_beta
        self.cluster_beta = None

    def fit(self, X):
        """
        Groups data into n_clusters using k-means algorithm
        Saves centroids to a BallTree

        Parameters
        ----------
        X : np.array
            Train transitions
        """
        super().fit(X)
        k_means = cluster.KMeans(n_clusters=self.n_clusters, init='k-means++', random_state=42)
        k_means = k_means.fit(X)
        non_empty = [i for i in range(self.n_clusters) if len(X[k_means.labels_ == i]) > 0]

        if self.use_beta:
            self.cluster_beta = \
                np.asarray([
                    self.dist(X[k_means.labels_ == i], k_means.cluster_centers_[i]).max()
                    for i in non_empty
                ]) * self.beta

    def pdf(self, transitions):
        """
        For a given transition, distance to nearest centroid is queried
        A probability density function is calculated as follows (1 / (distance_to_nearest_centroid + 1))

        Parameters
        ----------
        transitions : np.array
            Train transitions

        Returns
        -------
        float
            Probability density function
        """
        closest_centroid_dist, closest_centroid_id = self.centroid_tree.query(transitions, k=1)
        if self.use_beta:
            closest_centroid_dist = np.true_divide(
                closest_centroid_dist,
                self.cluster_beta[closest_centroid_id],
                out=np.ones_like(closest_centroid_dist, dtype=float),
                where=self.cluster_beta[closest_centroid_id] > 0)
            closest_centroid_dist = np.true_divide(
                closest_centroid_dist,
                self.cluster_beta[closest_centroid_id],
                out=np.ones_like(closest_centroid_dist),
                where=self.cluster_beta[closest_centroid_id] > 0
            )
        closest_centroid_dist = closest_centroid_dist.flatten()
        return 1 / (closest_centroid_dist + 1)


class KNNHyper(Detector):
    def fit(self, X):
        super().fit(X)
        def V(r, d):
            def gamma(n):
                if n == 1: return 1
                if n == 1 / 2: return np.sqrt(np.pi)
                return (n - 1) * gamma(n - 1)

            return np.pi ** d / gamma(d / 2 + 1) * r ** d
        self.h = self.scotts_factor
        self.V = V(self.h, X.shape[1])

    def pdf(self, transitions):
        test_probs = self.tree.query_radius(transitions, self.h, count_only=True) / (len(self.X) * self.V)
        return np.clip(test_probs, 0, 1)


class KNN(Detector):
    """
    KNN algorithm assumes that similar observations are in proximity to each other and outliers are usually
    lonely observations, staying farther from the cluster of similar observations

    ...

    Attributes
    ----------
    k : int
        Number of nearest neighbors to take into consideration
    f : function
        Function to apply over distances to knn

    Methods
    -------
    pdf(transitions)
        1, Distance to knn is calculated
        2, Function f is applied to those distances
        3, Probability density function is returned: (1 / (f result + 1) ). The closer the neighbors
        the higher probability of being considered as non-attack transition
    """
    def __init__(self, k=3, f=np.sum, **kwargs):
        super().__init__()
        self.k = k
        self.f = f

    def pdf(self, transitions):
        """
        1, Distance to knn is calculated
        2, Function f is applied to those distances
        3, Probability density function is returned: (1 / (f result + 1) ). The closer the neighbors
        the higher probability of being considered as non-attack transition

        Parameters
        ----------
        transitions : np.array
            Train transitions

        Returns
        -------
        float
            Probability density function
        """
        k_nearest_neighbors_distances = self.tree.query(transitions, k=self.k)[0]
        function_over_knn_distances = self.f(k_nearest_neighbors_distances, axis=1)
        return 1 / (function_over_knn_distances + 1)