#####
#
#   leaf wood separation from Vicari et al., 2019 without path tracing 
#

from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture as GMM

def set_nbrs_knn(arr, pts, knn, return_dist=True, block_size=100000):
    nbrs = NearestNeighbors(n_neighbors=knn, metric='euclidean',
                            algorithm='kd_tree', leaf_size=15,
                            n_jobs=-1).fit(arr)
    if block_size > pts.shape[0]:
        block_size = pts.shape[0]

    ids = np.arange(pts.shape[0])
    ids = np.array_split(ids, int(pts.shape[0] / block_size))

    if return_dist is True:
        distance = np.zeros([pts.shape[0], knn])
    indices = np.zeros([pts.shape[0], knn])

    if return_dist is True:
        
        for i in ids:
            nbrs_dist, nbrs_ids = nbrs.kneighbors(pts[i])
            distance[i] = nbrs_dist
            indices[i] = nbrs_ids
        return distance, indices

    elif return_dist is False:
        for i in ids:
            nbrs_ids = nbrs.kneighbors(pts[i], return_distance=False)
            indices[i] = nbrs_ids
        return indices
        

def classify(variables, n_classes):

    """
    Function to perform the classification of a dataset using sklearn's
    Gaussian Mixture Models with Expectation Maximization.
    Args:
        variables (array): N-dimensional array (m x n) containing a set of
             parameters (n) over a set of observations (m).
        n_classes (int): Number of classes to assign the input variables.
    Returns:
        classes (list): List of classes labels for each observation from the
            input variables.
        means (array): N-dimensional array (c x n) of each class (c) parameter
            space means (n).
        probability (array): Probability of samples belonging to every class
             in the classification. Sum of sample-wise probability should be
             1.
    """

    # Initialize a GMM classifier with n_classes and fit variables to it.
    gmm = GMM(n_components=n_classes)
    gmm.fit(variables)

    return gmm.predict(variables), gmm.means_, gmm.predict_proba(variables)

def class_select_abs(classes, cm, nbrs_idx, feature=5, threshold=0.5):

    """
    Select from GMM classification results which classes are wood and which
    are leaf based on a absolute value threshold from a single feature in
    the parameter space.
    Args:
        classes (list or array): Classes labels for each observation from the
            input variables.
        cm (array): N-dimensional array (c x n) of each class (c) parameter
            space mean valuess (n).
        nbrs_idx (array): Nearest Neighbors indices relative to every point
            of the array that originated the classes labels.
        feature (int): Column index of the feature to use as constraint.
        threshold (float): Threshold value to mask classes. All classes with
            means >= threshold are masked as true.
    Returns:
        mask (list): List of booleans where True represents wood points and
            False represents leaf points.
    """

    # Calculating the ratio of first 3 components of the classes means (cm).
    # These components are the basic geometric descriptors.
    if np.max(np.sum(cm, axis=1)) >= threshold:

        class_id = np.argmax(cm[:, feature])

        # Masking classes based on the criterias set above. Mask will present
        # True for wood points and False for leaf points.
        mask = classes == class_id

    else:
        mask = []

    return mask

def wlseparate_abs(arr, knn, knn_downsample=1, n_classes=3):

    """
    Classifies a point cloud (arr) into three main classes, wood, leaf and
    noclass.
    The final class selection is based on the absolute value of the last
    geometric feature (see point_features module).
    Points will be only classified as wood or leaf if their classification
    probability is higher than prob_threshold. Otherwise, points are
    assigned to noclass.
    Class selection will mask points with feature value larger than a given
    threshold as wood and the remaining points as leaf.
    Args:
        arr (array): Three-dimensional point cloud of a single tree to perform
            the wood-leaf separation. This should be a n-dimensional array
            (m x n) containing a set of coordinates (n) over a set of points
            (m).
        knn (int): Number of nearest neighbors to search to constitue the
            local subset of points around each point in 'arr'.
        knn_downsample (float): Downsample factor (0, 1) for the knn
            parameter. If less than 1, a sample of size (knn * knn_downsample)
            will be selected from the nearest neighbors indices. This option
            aims to maintain the spatial representation of the local subsets
            of points, but reducing overhead in memory and processing time.
        n_classes (int): Number of classes to use in the Gaussian Mixture
            Classification.
    Returns:
        class_indices (dict): Dictionary containing indices for wood and leaf
            classes.
        class_probability (dict): Dictionary containing probabilities for wood
            and leaf classes.
    """

    # Generating the indices array of the 'k' nearest neighbors (knn) for all
    # points in arr.
    idx_1 = set_nbrs_knn(arr, arr, knn, return_dist=False)

    # If downsample fraction value is set to lower than 1. Apply downsampling
    # on knn indices.
    if knn_downsample < 1:
        n_samples = np.int(idx_1.shape[1] * knn_downsample)
        idx_f = np.zeros([idx_1.shape[0], n_samples + 1])
        idx_f[:, 0] = idx_1[:, 0]
        for i in range(idx_f.shape[0]):
            idx_f[i, 1:] = np.random.choice(idx_1[i, 1:], n_samples,
                                            replace=False)
        idx_1 = idx_f.astype(int)

    # Calculating geometric descriptors.
    gd_1 = knn_features(arr, idx_1)

    # Classifying the points based on the geometric descriptors.
    classes_1, cm_1, proba_1 = classify(gd_1, n_classes)

    # Selecting which classes represent wood and leaf. Wood classes are masked
    # as True and leaf classes as False.
    mask_1 = class_select_abs(classes_1, cm_1, idx_1)

    # Generating set of indices of entries in arr. This will be part of the
    # output.
    arr_ids = np.arange(0, arr.shape[0], 1, dtype=int)

    # Creating output class indices dictionary.
    # mask represent wood points, (~) not mask represent leaf points.
    class_indices = {}
    class_indices['wood'] = arr_ids[mask_1]
    class_indices['leaf'] = arr_ids[~mask_1]

    # Creating output class probabilities dictionary.
    # mask represent wood points, (~) not mask represent leaf points.
    class_probability = {}
    class_probability['wood'] = np.max(proba_1, axis=1)[mask_1]
    class_probability['leaf'] = np.max(proba_1, axis=1)[~mask_1]

    return class_indices, class_probability