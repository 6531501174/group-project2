# animal_clustering.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# โหลดข้อมูลจากไฟล์
with open("name of the animals.txt", "r") as f:
    animals = [line.strip() for line in f if line.strip()]

# แปลงชื่อสัตว์ให้เป็นเวกเตอร์ด้วย TF-IDF (ใช้ character n-grams)
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
X = vectorizer.fit_transform(animals)

# ทำ Clustering ด้วย KMeans
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(X)

# สร้าง DataFrame
df = pd.DataFrame({'Animal': animals, 'Cluster': labels})

# ลดมิติสำหรับ visualization ด้วย PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

# บันทึกผลลัพธ์
df.to_csv("animal_clusters.csv", index=False)
print("Clustering complete. File saved as 'animal_clusters.csv'")
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

