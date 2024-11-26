import matplotlib.pyplot as plt
import numpy as np

# Example data (replace with your actual values)
classes = np.arange(149)  # Class indices
f1_scores = np.random.uniform(0, 1, 149)  # Replace with actual F1-scores
class_counts = np.random.randint(10, 150, 149)  # Replace with class sample counts

# Bar chart: F1-scores
plt.figure(figsize=(10, 6))
colors = ['green' if f > 0.6 else 'red' if f < 0.2 else 'gray' for f in f1_scores]
plt.bar(classes, f1_scores, color=colors)
plt.title('Class-wise F1-Scores')
plt.xlabel('Class Index')
plt.ylabel('F1-Score')
plt.axhline(y=0.6, color='green', linestyle='--', label='High Performance Threshold')
plt.axhline(y=0.2, color='red', linestyle='--', label='Low Performance Threshold')
plt.legend()
plt.show()

# Histogram: Class Distribution
plt.figure(figsize=(10, 6))
plt.bar(classes, class_counts, color='blue')
plt.axhline(y=np.mean(class_counts), color='orange', linestyle='--', label='Average Class Size')
plt.title('Class Distribution')
plt.xlabel('Class Index')
plt.ylabel('Sample Count')
plt.legend()
plt.show()