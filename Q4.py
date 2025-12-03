import numpy as np
import matplotlib.pyplot as plt

def run_kmeans():
    # --- Configuration ---
    filename = "Q4_dataset.txt"
    rows = 516
    cols = 407
    K = 8
    
    # 1. Load Data
    print(f"Loading data from {filename}...")
    try:
        # Assuming the file is space or comma separated RGB values
        X = np.loadtxt(filename)
    except OSError:
        print(f"Error: {filename} not found. Please make sure the file is in the same directory.")
        return

    # Ensure data is the right shape (N, 3)
    if X.shape[1] != 3:
        print("Error: Dataset should have 3 columns (R, G, B).")
        return
        
    print(f"Data loaded: {X.shape[0]} pixels.")

    # 2. Define Initial Centroids
    centroids = np.array([
        [255, 255, 255], 
        [255, 0, 0],     
        [128, 0, 0],     
        [0, 255, 0],     
        [0, 128, 0],     
        [0, 0, 255],     
        [0, 0, 128],     
        [0, 0, 0]        
    ], dtype=np.float64)

    print("\nStarting K-Means clustering...")
    
    iteration = 0
    max_iterations = 50 # Safety break, though usually converges faster
    sse_history = []
    
    # Store cluster assignments for every pixel (initialized to -1)
    labels = np.zeros(X.shape[0], dtype=int)
    
    while iteration < max_iterations:
        iteration += 1
        
        # --- Step 1: Assignment --- # 
        # Distances shape: (num_pixels, K)
        
        # Expand X to (N, 1, 3) and centroids to (1, K, 3) to broadcast subtract
        distances = np.sqrt(((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
        
        # Get new assignments (index of the nearest centroid)
        new_labels = np.argmin(distances, axis=1)
        
        # Calculate Sum of Squared Errors (Distances) for this iteration
        # We take the distance to the chosen centroid for each pixel
        min_distances = np.min(distances, axis=1)
        sse = np.sum(min_distances ** 2)
        sse_history.append(sse)
        
        # --- Step 2: Update Centroids ---
        new_centroids = np.zeros_like(centroids)
        active_clusters = 0
        
        # Check for convergence (if labels didn't change, centroids won't change)
        if np.array_equal(labels, new_labels):
            print(f"Converged at iteration {iteration}!")
            break
            
        labels = new_labels

        pixel_counts = []
        
        for k in range(K):
            # Find all points assigned to cluster k
            points_in_cluster = X[labels == k]
            
            if len(points_in_cluster) > 0:
                # Calculate mean
                new_centroids[k] = points_in_cluster.mean(axis=0)
                active_clusters += 1
                pixel_counts.append(len(points_in_cluster))
            else:
                # Cluster "disappeared" - keep old centroid or reset. 
                # Assignment implies it just disappears from being updated.
                new_centroids[k] = centroids[k] 
                pixel_counts.append(0)

        centroids = new_centroids
        
        print(f"Iteration {iteration}: SSE = {sse:.2f}")

    #RESULTS
    print("\n" + "="*30)
    print("FINAL REPORT")
    print("="*30)
    
    # 1. Active Clusters
    active_count = np.sum(np.array(pixel_counts) > 0)
    print(f"1. Total active clusters at the end: {active_count}")
    
    # 2. Final Centroids
    print("\n2. Final Centroids (R, G, B):")
    for k in range(K):
        print(f"   Cluster {k+1}: {centroids[k].round(2)}")

    # 3. Pixels per cluster
    print("\n3. Number of pixels associated to each cluster:")
    for k in range(K):
        print(f"   Cluster {k+1}: {pixel_counts[k]}")

    # 4. SSE History
    print("\n4. SSE (Sum of Squared Distances) per iteration:")
    for i, val in enumerate(sse_history):
        print(f"   Iter {i+1}: {val:.2f}")

    # --- Visualization ---
    print("\nGenerating visualization...")
    
    # Replace every pixel with its centroid color
    # Create an array of shape (N, 3)
    reconstructed_flat = centroids[labels]
    
    # Reshape back to image dimensions (Rows, Cols, Channels)
    # We must cast to uint8 for imshow to interpret as 0-255 image data correctly
    reconstructed_image = reconstructed_flat.reshape(rows, cols, 3).astype(np.uint8)
    
    plt.figure(figsize=(8, 10))
    plt.imshow(reconstructed_image)
    plt.title(f'K-Means Result (K={K})')
    plt.axis('off')
    plt.savefig("Q4_Visualization.png")
    plt.show()

if __name__ == "__main__":
    run_kmeans()