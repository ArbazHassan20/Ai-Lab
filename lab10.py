import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("Mall_Customers.csv")

# Drop CustomerID and isolate features
X_all = df.drop("CustomerID", axis=1)

# ------- Clustering without scaling -------
kmeans_no_scaling = KMeans(n_clusters=5, random_state=42)
df['Cluster_No_Scaling'] = kmeans_no_scaling.fit_predict(X_all)

# ------- Clustering with scaling (except Age) -------
X_scaled = X_all.copy()
scaler = StandardScaler()
X_scaled[['Annual Income (k$)', 'Spending Score (1-100)']] = scaler.fit_transform(X_scaled[['Annual Income (k$)', 'Spending Score (1-100)']])

kmeans_scaled = KMeans(n_clusters=5, random_state=42)
df['Cluster_With_Scaling'] = kmeans_scaled.fit_predict(X_scaled)

print(df[['CustomerID', 'Cluster_No_Scaling', 'Cluster_With_Scaling']].head())

# Optional: Visualize both clusterings
plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
plt.title("Without Scaling")
plt.scatter(X_all['Annual Income (k$)'], X_all['Spending Score (1-100)'], c=df['Cluster_No_Scaling'])

plt.subplot(1, 2, 2)
plt.title("With Scaling")
plt.scatter(X_all['Annual Income (k$)'], X_all['Spending Score (1-100)'], c=df['Cluster_With_Scaling'])
plt.show()


Task 2:
data = {
    'vehicle_serial_no': [5, 3, 8, 2, 4, 7, 6, 10, 1, 9],
    'mileage': [150000, 120000, 250000, 80000, 100000, 220000, 180000, 300000, 75000, 280000],
    'fuel_efficiency': [15, 18, 10, 22, 20, 12, 16, 8, 24, 9],
    'maintenance_cost': [5000, 4000, 7000, 2000, 3000, 6500, 5500, 8000, 1500, 7500],
    'vehicle_type': ['SUV', 'Sedan', 'Truck', 'Hatchback', 'Sedan', 'Truck', 'SUV', 'Truck', 'Hatchback', 'SUV']
}

df = pd.DataFrame(data)

# Remove categorical & ID column
X = df.drop(['vehicle_serial_no', 'vehicle_type'], axis=1)

# Clustering without scaling
kmeans_ns = KMeans(n_clusters=3, random_state=42)
df['Cluster_No_Scaling'] = kmeans_ns.fit_predict(X)

# Clustering with scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans_s = KMeans(n_clusters=3, random_state=42)
df['Cluster_With_Scaling'] = kmeans_s.fit_predict(X_scaled)

print(df[['vehicle_serial_no', 'Cluster_No_Scaling', 'Cluster_With_Scaling']])


Task 3:
import numpy as np

# Sample student data
data = {
    'student_id': range(1, 11),
    'GPA': [2.5, 3.2, 3.8, 2.0, 3.5, 2.8, 3.0, 3.9, 1.9, 3.7],
    'study_hours': [8, 12, 15, 5, 14, 9, 10, 16, 4, 13],
    'attendance_rate': [70, 85, 90, 60, 88, 75, 80, 95, 55, 92]
}
df = pd.DataFrame(data)

# Select and scale features
features = df[['GPA', 'study_hours', 'attendance_rate']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Find optimal K using Elbow method
wcss = []
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(features_scaled)
    wcss.append(km.inertia_)

# Plot Elbow
plt.plot(range(2, 7), wcss, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.show()

# From elbow, let's assume K=3
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(features_scaled)

# Show clusters
print(df[['student_id', 'Cluster']])

# Visualize
plt.scatter(df['study_hours'], df['GPA'], c=df['Cluster'], cmap='rainbow')
plt.xlabel('Study Hours')
plt.ylabel('GPA')
plt.title('Student Clustering by GPA & Study Hours')
plt.grid(True)
plt.show()
