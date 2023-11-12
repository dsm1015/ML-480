import numpy as np
from scipy.spatial.distance import euclidean
from queue import Queue
import time

class ClusteringFeature:
    def __init__(self, data_point):
        self.n = 1  # Number of data points
        self.linear_sum = np.array(data_point)  # Linear sum of data points
        self.squared_sum = np.sum(np.square(data_point))  # Squared sum for computing variance
        self.children = []
        self.data_points = [data_point]

    def update(self, data_point):
        self.n += 1
        self.linear_sum += data_point
        self.squared_sum += np.sum(np.square(data_point))
        self.data_points.append(data_point)

    def centroid(self):
        return self.linear_sum / self.n

    def radius(self):
        return np.sqrt(self.squared_sum / self.n - np.square(self.centroid()))

    def distance(self, data_point):
        return euclidean(self.centroid(), data_point)


class CFNode:
    def __init__(self, branching_factor, threshold, is_leaf=True):
        self.branching_factor = branching_factor
        self.threshold = threshold
        self.is_leaf = is_leaf
        self.children = []  # List of ClusteringFeatures or CFNodes
        self.parent = None

    def insert(self, data_point):
        if not self.children:
            new_cf = ClusteringFeature(data_point)
            self.children.append(new_cf)
            return

        closest_child, closest_cf, min_distance = self.find_closest_child(data_point)

        if min_distance <= self.threshold**2:
            closest_cf.update(data_point)
        else:
            new_cf = ClusteringFeature(data_point)
            self.children.append(new_cf)
            if len(self.children) > self.branching_factor:
                self.split_node()

    def find_closest_child(self, data_point):
        closest_child = None
        closest_cf = None
        min_distance = float('inf')
        for child in self.children:
            if isinstance(child, ClusteringFeature):
                dist = child.distance(data_point)
                cf = child
            else:  # if child is a CFNode
                cf, dist = child.find_closest_cf_and_distance(data_point)

            if dist < min_distance:
                min_distance = dist
                closest_child = child
                closest_cf = cf
        return closest_child, closest_cf, min_distance

    def find_closest_cf_and_distance(self, data_point):
        if self.is_leaf:
            return self.find_closest_child(data_point)
        else:
            closest_cf = None
            min_distance = float('inf')
            for child in self.children:
                cf, dist = child.find_closest_cf_and_distance(data_point)
                if dist < min_distance:
                    min_distance = dist
                    closest_cf = cf
            return closest_cf, min_distance

    def split_node(self):
        pass  # Placeholder for actual splitting logic

    def get_clusters(self):
        if self.is_leaf:
            return self.children
        else:
            clusters = []
            for child in self.children:
                clusters.extend(child.get_clusters())
            return clusters


class BIRCH:
    def __init__(self, branching_factor, threshold, max_memory_limit=None):
        self.root = CFNode(branching_factor, threshold)
        self.max_memory_limit = max_memory_limit * 1024 * 1024 if max_memory_limit else None
        self.current_memory_usage = 0
        self.max_threshold_factor = 2
        self.initial_threshold = threshold
        self.max_threshold = threshold * self.max_threshold_factor
        self.threshold_adjustments = []

    def fit(self, data):
        self.start_time = time.time()
        for data_point in data:
            self.current_memory_usage += data_point.nbytes
            # Check if we've exceeded the memory limit
            if self.max_memory_limit and self.current_memory_usage > self.max_memory_limit:
                # Adjust the threshold to control the size of the CF-Tree
                if self.root.threshold < self.max_threshold:
                    self.root.threshold = self.adjust_threshold()
            # Find the closest leaf node and insert the data point
            closest_leaf = self.find_closest_leaf(self.root, data_point)
            closest_leaf.insert(data_point)
        self.total_time = time.time() - self.start_time

    def find_closest_leaf(self, node, data_point):
        queue = Queue()
        queue.put(node)
        while not queue.empty():
            current_node = queue.get()
            if current_node.is_leaf:
                return current_node
            else:
                closest_child, _, _ = current_node.find_closest_child(data_point)
                queue.put(closest_child)
    
    def adjust_threshold(self):
        current_time = time.time()
        adjustment_factor = self.current_memory_usage / self.max_memory_limit
        adjusted_threshold = self.root.threshold * adjustment_factor
        adjusted_threshold = min(max(adjusted_threshold, self.root.threshold), self.max_threshold)
        relative_time = current_time - self.start_time
        self.threshold_adjustments.append((relative_time, adjusted_threshold))
        return adjusted_threshold
    
    def predict(self, X):
        labels = []
        self.get_clusters()
        for data_point in X:
            closest_leaf = self.find_closest_leaf(self.root, data_point)
            closest_cf, _ , _ = closest_leaf.find_closest_cf_and_distance(data_point)
            labels.append(closest_cf.label)
        return labels
    
    def get_clusters(self):
        clusters = self.root.get_clusters()
        # Assign a unique label to each ClusteringFeature
        for i, cf in enumerate(clusters):
            cf.label = i
        return clusters

"""
from sklearn.datasets import make_blobs
from sklearn.datasets import load_iris
from sklearn.cluster import Birch


dataset, blob_clusters = make_blobs(n_samples = 100, n_features=2, centers=10, random_state = 48)
#dataset = load_iris().data

sklearn_birch = Birch(branching_factor=50, threshold=4, n_clusters=None)
start_time = time.time()
sklearn_birch.fit(dataset)
end_time = time.time()
sklearn_time = end_time - start_time
pred = sklearn_birch.predict(dataset)

# Instantiate BIRCH algorithm with a fixed threshold
birch = BIRCH(branching_factor=50, threshold=4, max_memory_limit=0.5)
start_time = time.time()
birch.fit(dataset)
end_time = time.time()
my_time = end_time - start_time

# Get the centroids of the clusters
clusters = birch.get_clusters()
my_labels = birch.predict(dataset)

# Measure time for scikit-learn Birch
print(f"Scikit-learn Birch runtime: {sklearn_time} seconds")
print(f"My Birch runtime: {my_time} seconds")


print(f"Number of clusters in scikit-learn Birch: {len(np.unique(pred))}")
print(f"Number of clusters in my Birch: {len(clusters)}")


import matplotlib.pyplot as plt


# Helper function to calculate the centroids of clusters for custom BIRCH implementation
def calculate_custom_birch_centroids(clusters):
    return [np.mean(np.array(cf.data_points), axis=0) for cf in clusters]

# Calculate centroids for custom BIRCH
custom_birch_centroids = calculate_custom_birch_centroids(clusters)
custom_birch_centroids = np.array(custom_birch_centroids)  # Convert to NumPy array for consistent indexing

# Calculate centroids for scikit-learn Birch
sklearn_birch_centroids = sklearn_birch.subcluster_centers_

# Plotting for visual comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=True)

# Scikit-learn Birch clusters
axes[0,0].scatter(dataset[:, 0], dataset[:, 1], c=pred, cmap='rainbow', alpha=0.7, edgecolors='b')
axes[0,0].scatter(sklearn_birch_centroids[:, 0], sklearn_birch_centroids[:, 1], c='red', marker='x', label='Centroids')
axes[0,0].set_title('Scikit-learn Birch Clusters')
axes[0,0].set_xlabel('Feature 1')
axes[0,0].set_ylabel('Feature 2')

# Custom Birch clusters
axes[0,1].scatter(dataset[:, 0], dataset[:, 1], c=my_labels, cmap='rainbow', alpha=0.7, edgecolors='b')
axes[0,1].scatter(custom_birch_centroids[:, 0], custom_birch_centroids[:, 1], c='red', marker='x')  # No label argument here
axes[0,1].set_title('Custom Birch Clusters')
axes[0,1].set_xlabel('Feature 1')

adjustment_times, thresholds = zip(*birch.threshold_adjustments)

relative_times_fraction = [t / birch.total_time for t in adjustment_times]
axes[1,0].plot(relative_times_fraction, thresholds, marker='o')
axes[1,0].axhline(y=birch.initial_threshold, color='r', linestyle='--', label='Initial Threshold')
axes[1,0].axhline(y=birch.max_threshold, color='g', linestyle='--', label='Max Threshold')
axes[1,0].grid()
axes[1,0].set_ylim([0, birch.max_threshold * 1.1])
axes[1,0].set_title('Threshold Adjustments')
axes[1,0].set_xlabel('Time')
axes[1,0].set_ylabel('Threshold')

plt.tight_layout()
plt.show()
"""