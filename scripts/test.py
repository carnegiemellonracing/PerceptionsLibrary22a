import numpy as np

# Example matrices
random_matrix = np.random.rand(3, 3)  # Replace with your actual random matrix
group_matrix = np.array([[1, 1, 2], [2, 2, 1], [1, 2, 1]])  # Replace with your actual group matrix

# Get unique group labels
unique_groups = np.unique(group_matrix)

# Create a 3D array to store split matrices
split_matrices = np.zeros((len(unique_groups), *random_matrix.shape))

# Use advanced indexing to assign values to split_matrices
split_matrices[unique_groups - 1, :, :] = random_matrix

# Print the result
for group, matrix in zip(unique_groups, split_matrices):
    print(f"Group {group} Matrix:")
    print(matrix)
    print()