# Clustering

# Customer Segmentation using K-Means Clustering  
**With Jupyter Notebook**

This Jupyter Notebook demonstrates how to perform customer segmentation using K-Means clustering on the Mall Customers dataset. The dataset contains information about customers, including their `Age`, `Annual Income (k$)`, and `Spending Score (1-100)`. The goal is to group customers into clusters based on their spending behavior and income.

---

## **Table of Contents**
1. [Prerequisites](#prerequisites)
2. [Getting Started](#getting-started)
3. [Running the Code](#running-the-code)
4. [Code Explanation](#code-explanation)
5. [Results](#results)
6. [License](#license)

---

## **Prerequisites**
Before running the code, ensure you have the following installed:
- Python 3.x
- Required Python libraries:
  ```bash
  pip install numpy pandas matplotlib seaborn scikit-learn plotly jupyter
  ```
- Jupyter Notebook (to run the `.ipynb` file).

---

## **Getting Started**
1. **Download the Dataset**  
   Ensure the dataset `Mall_Customers.csv` is in the same directory as the notebook.

2. **Launch Jupyter Notebook**  
   Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
   Open the `.ipynb` file from the Jupyter Notebook interface.

---

## **Running the Code**
1. Open the `.ipynb` file in Jupyter Notebook.
2. Run each cell sequentially to execute the code.

---

## **Code Explanation**
### **1. Import Libraries**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
%matplotlib inline
```
- Libraries used for data manipulation, visualization, and clustering.

### **2. Load and Explore Data**
```python
Mall_data = pd.read_csv('Mall_Customers.csv')
Mall_data.head()
Mall_data.describe()
Mall_data.info()
```
- Load the dataset and explore its structure, summary statistics, and data types.

### **3. Data Visualization**
```python
sns.countplot(data=Mall_data, x='Gender')
plt.figure(figsize=(5, 5))
plt.pie(Mall_data['Gender'].value_counts(), labels=['FEMALE', 'MALE'], autopct='%.1f%%', colors=['green', 'orange'])
plt.show()
```
- Visualize the distribution of gender using a count plot and pie chart.

### **4. Feature Distribution**
```python
figure, axes = plt.subplots(1, 3, figsize=(12, 5))
sns.histplot(data=Mall_data, x='Annual Income (k$)', hue='Gender', kde=True, ax=axes[0])
sns.histplot(data=Mall_data, x='Spending Score (1-100)', hue='Gender', kde=True, ax=axes[1])
sns.histplot(data=Mall_data, x='Age', hue='Gender', kde=True, ax=axes[2])
plt.show()
```
- Visualize the distribution of `Annual Income`, `Spending Score`, and `Age` using histograms.

### **5. Pairplot**
```python
sns.pairplot(data=Mall_data, x_vars=["Age", 'Annual Income (k$)', 'Spending Score (1-100)'],
             y_vars=["Age", 'Annual Income (k$)', 'Spending Score (1-100)'], hue='Gender', diag_kind='hist')
```
- Visualize relationships between features using a pairplot.

### **6. K-Means Clustering (2D)**
```python
data1 = Mall_data.iloc[:, [3, 4]]  # Annual Income and Spending Score
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(data1)
    sse.append(kmeans.inertia_)
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()
```
- Determine the optimal number of clusters using the Elbow Method.

### **7. Apply K-Means Clustering**
```python
kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
kmeans.fit(data1)
print("SSE:", kmeans.inertia_)
print("Centroid:", kmeans.cluster_centers_)
print("Iteration:", kmeans.n_iter_)
```
- Apply K-Means clustering with 5 clusters.

### **8. Visualize Clusters (2D)**
```python
plt.figure(figsize=(12, 7))
plt.scatter(data1['Annual Income (k$)'], data1['Spending Score (1-100)'], s=30, c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=70)
plt.title('K Means Clustering', fontsize=20)
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.show()
```
- Visualize the clusters in 2D.

### **9. K-Means Clustering (3D)**
```python
data2 = Mall_data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(data2)
    sse.append(kmeans.inertia_)
plt.figure(figsize=(12, 5))
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()
```
- Determine the optimal number of clusters for 3D clustering.

### **10. Apply K-Means Clustering (3D)**
```python
kmeans = KMeans(n_clusters=6, n_init=10, random_state=42)
kmeans.fit(data2)
print("SSE:", kmeans.inertia_)
print("Centroid:", kmeans.cluster_centers_)
print("Iteration:", kmeans.n_iter_)
```
- Apply K-Means clustering with 6 clusters.

### **11. Visualize Clusters (3D)**
```python
fig = px.scatter_3d(
    data2,
    x='Age',
    y='Annual Income (k$)',
    z='Spending Score (1-100)',
    color=kmeans.labels_.astype(str),
    opacity=0.8,
    height=700,
    width=700,
    title="Clusters in Age-Annual Income-Spending Score",
    color_discrete_sequence=px.colors.qualitative.Set2)
fig.show()
```
- Visualize the clusters in 3D using Plotly.

---

## **Results**
- **Optimal Number of Clusters**: Determined using the Elbow Method.
- **Cluster Visualization**: 2D and 3D visualizations of customer segments.
- **Cluster Centroids**: Centers of each cluster for interpretation.

---

## **License**
This project is open-source and available under the [MIT License](LICENSE). Feel free to use, modify, and distribute it as needed.

---

## **Support**
If you encounter any issues or have questions, feel free to open an issue in this repository or contact me at [minthukywe2020@gmail.com](mailto:minthukywe2020@gmail.com).

---

Enjoy exploring customer segmentation using K-Means clustering in Jupyter Notebook! 🚀
