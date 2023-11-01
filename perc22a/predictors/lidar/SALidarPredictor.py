import numpy as np
from sklearn.cluster import DBSCAN

class LidarPredictor():
    def __init__(self):
        self.data = np([])

    def predict(self, data):
        self.data = data["point cloud"]

        preprocessed_data = self.preprocess_lidar_data(data)
        
        clusters = self.cluster_lidar_data(preprocessed_data)
        
        # Identify cones in clusters and categorize them
        orange_cones, blue_cones, yellow_cones = self.detect_and_categorize_cones(clusters)

        return orange_cones, blue_cones, yellow_cones

    def preprocess_lidar_data(self, data):
        arr = np([])
        for i in data:
            x, y, z, intensity, ring, timestamp = data[i]
            if intensity > 100:
                np.append(arr, data[i])
        return arr
    
    def cluster_lidar_data(self, data):
        clustering = DBSCAN(eps=0.2, min_samples=5)
        labels = clustering.fit_predict(data)
        unique_labels = np.unique(labels)
        clusters = [data[labels == label] for label in unique_labels if label != -1]
        return clusters
    
    def detect_and_categorize_cones(self, clusters):
        orange_cones = []
        blue_cones = []
        yellow_cones = []

        for cluster in clusters:
            #classify clusters based on height (maybe not best method )
            z_values = cluster[:, 2]
            cluster_height = np.max(z_values) - np.min(z_values)

            if cluster_height > 0.3:
                if np.mean(cluster[:, 3]) > 150:
                    orange_cones.append(cluster)
                else:
                    blue_cones.append(cluster)
            else:
                yellow_cones.append(cluster)
        
        return orange_cones, blue_cones, yellow_cones

    
        


if __name__ == '__main__':
    # Example usage
    from perc22a.data.utils.DataLoader import DataLoader  # Import DataLoader or adapt as needed

    dl = DataLoader('./data/raw/track-testing-09-29/')  # Replace '<path>' with the actual data path
    predictor = LidarPredictor()
    
    for i in range(len(dl)):
        data = dl[i]
        orange_cones, blue_cones, yellow_cones = predictor.predict(data)

