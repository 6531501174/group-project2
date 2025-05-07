# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# โหลดข้อมูลคลัสเตอร์
df = pd.read_csv("animal_clusters.csv")

st.title("🐾 Animal Name Clustering")
st.markdown("แอปนี้แสดงผลการจัดกลุ่มชื่อสัตว์ด้วยการวิเคราะห์รูปแบบตัวอักษร (n-gram)")

# ค้นหาสัตว์
name = st.text_input("🔍 ค้นหาชื่อสัตว์:")
if name:
    result = df[df['Animal'].str.lower() == name.lower()]
    if not result.empty:
        st.success(f"'{name}' อยู่ใน Cluster หมายเลข {result.iloc[0]['Cluster']}")
    else:
        st.error("ไม่พบชื่อสัตว์ในรายการ")

# แสดงสัตว์ในคลัสเตอร์ที่เลือก
cluster = st.selectbox("📂 เลือก Cluster เพื่อดูชื่อสัตว์:", sorted(df['Cluster'].unique()))
st.dataframe(df[df['Cluster'] == cluster][['Animal']])

# แสดงกราฟ Visualization
st.subheader("📊 Visualization ของคลัสเตอร์")
fig, ax = plt.subplots()
for i in df['Cluster'].unique():
    cluster_data = df[df['Cluster'] == i]
    ax.scatter(cluster_data['PCA1'], cluster_data['PCA2'], label=f"Cluster {i}")
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.set_title("PCA Clustering Visualization")
ax.legend()
st.pyplot(fig)
# -*- coding: utf-8 -*-


