#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import ee

from os import listdir
from os.path import join
import os


# In[3]:


ee.Authenticate()
ee.Initialize()


# In[4]:


startDate = pd.to_datetime('2000-08-01')
endDate = pd.to_datetime('2023-08-01')
dates_list = pd.date_range(start='2000-08-01', freq='AS-AUG', periods=24)

lat, lon = 45,60
point = ee.Geometry.Point([lon, lat]).buffer(distance=5000)


# In[5]:


IMGS = list()
for date in dates_list:
  MOD = ee.ImageCollection("MODIS/061/MOD09GA")\
                .filterDate(start = date)\
                .filterBounds(point)
  IMGS.append(ee.Image(MOD.first()))


# In[6]:


get_ipython().system('pip install geemap')


# In[37]:


import geemap


# In[38]:


bands = ['sur_refl_b02', 'sur_refl_b04', 'sur_refl_b03']


# In[39]:


map = geemap.Map(center=[lat, lon], zoom=8)
roi = ee.Geometry.Rectangle([57, 43,62, 47])
pars = {'min': -100.0,
  'max': 8000.0,}
map.add_ee_layer(IMGS[0].select(bands).clip(roi), pars)
print(map)


# In[40]:


map


# In[42]:


col = geemap.create_timeseries(
        ee.ImageCollection(IMGS),
        '2000-08-01',
        '2023-08-01',
        region=roi,
        bands=bands,
    )
col


# In[43]:


col = col.select(bands).map(
        lambda img: img.visualize(**pars).set(
            {
                "system:time_start": img.get("system:time_start"),
                "system:date": img.get("system:date"),
            }
        )
    )


# In[33]:


video_args = {}
video_args["dimensions"] = 768
video_args["region"] = roi
video_args["framesPerSecond"] = 30
video_args["crs"] = "EPSG:3857"
video_args["min"] = 0
video_args["max"] = 255
video_args["bands"] = ["vis-red", "vis-green", "vis-blue"]


# In[44]:


out_dir='/Users/ch.bharathchandra/Desktop/datasets'
count = col.size().getInfo()
basename = 'MOD'
names = [
    os.path.join(
        out_dir, f"{basename}_{str(i+1).zfill(int(len(str(count))))}.jpg"
    )
    for i in range(count)
]
geemap.get_image_collection_thumbnails(
            col,
            './MOD',
            vis_params={
                "min": 0,
                "max": 255,
                "bands": video_args["bands"],
            },
            dimensions=768,
            names=names,
        )

geemap.make_gif(
    ['./MOD/'+x for x in names],
    'MOD.gif',
    fps=2,
    mp4=False,
    clean_up=True,
)

geemap.add_text_to_gif(
  'MOD.gif',
  'MOD.gif',
  text_sequence=[x.strftime('%Y-%m-%d') for x in dates_list],
  font_type='monospace',
  font_size=24,
  font_color='white',
  duration=1000 / 3,

)


# In[22]:


import matplotlib.image as mpimg


# In[23]:


img = mpimg.imread('/Users/ch.bharathchandra/Desktop/datasets/MOD_03.jpg')
img2 = mpimg.imread('/Users/ch.bharathchandra/Desktop/datasets/MOD_18.jpg')

fig, ax = plt.subplots(ncols=2, figsize=(16,9))
ax[0].imshow(img)
ax[1].imshow(img2)
for i in range(2):
  ax[i].set_facecolor('black')
  ax[i].set_xticks([])
  ax[i].set_yticks([])
ax[0].set_title('2000-08-01', fontsize=26)
ax[1].set_title('2023-08-01', fontsize=26)
plt.show()


# In[45]:


'''Zooming in'''
img = img[140:600,110:500,:]
img2 = img2[140:600,110:500,:]

fig, ax = plt.subplots(ncols=2, figsize=(16,9))
ax[0].imshow(img)
ax[1].imshow(img2)
for i in range(2):
  ax[i].set_facecolor('black')
  ax[i].set_xticks([])
  ax[i].set_yticks([])
ax[0].set_title('2000-08-01', fontsize=26)
ax[1].set_title('2023-08-01', fontsize=26)
plt.show()


# In[46]:


df = pd.DataFrame({'R': img[:,:, 0].flatten(), 'G': img[:,:, 1].flatten(), 'B':img[:,:, 2].flatten()})
df2 = pd.DataFrame({'R': img2[:,:, 0].flatten(), 'G': img2[:,:, 1].flatten(), 'B':img2[:,:, 2].flatten()})


# In[48]:


def distance(data, centroids):
    """
    Calculates the Euclidean distance between data points and centroids.

    Args:
        data (pd.DataFrame): Dataframe containing data points with R, G, B columns.
        centroids (np.array): Array of centroids with shape (K, 3).

    Returns:
        pd.DataFrame: Dataframe with an added 'Class' column assigning points to clusters.
    """
    cols = list()
    for i in range(1, len(centroids) + 1):
        data[f'C{i}'] = ((centroids[i - 1][0] - data.R) ** 2 +
                          (centroids[i - 1][1] - data.G) ** 2 +
                          (centroids[i - 1][2] - data.B) ** 2) ** 0.5
    cols.extend([f'C{i}' for i in range(1, len(centroids) + 1)])
    data['Class'] = data[cols].abs().idxmin(axis=1)
    return data


def kmeans(data, K):
    """
    Performs k-means clustering on the data with Euclidean distance.

    Args:
        data (pd.DataFrame): Dataframe containing data points with R, G, B columns.
        K (int): Number of clusters.

    Returns:
        tuple: A tuple containing the clustered data and a list of distances at each iteration.
    """
    print(10 * '-', f'k={K}\tDistance=Euclidean', '-' * 10)
    L = list()
    new_centroids = data.sample(K).values

    data = distance(data.copy(), new_centroids)
    old_centroids = new_centroids.copy()
    new_centroids = np.array([data[data.Class == Class][['R', 'G', 'B']].mean().values for Class in data.loc[:,'C1':f'C{K}'].columns])
    i = 1
    print(f'Iteration: {i}\tDistance: {abs(new_centroids.mean() - old_centroids.mean())}')
    while abs(new_centroids.mean() - old_centroids.mean()) > 0.001:
        L.append(abs(new_centroids.mean() - old_centroids.mean()))
        data = distance(data, new_centroids)
        old_centroids = new_centroids.copy()
        new_centroids = np.array([data[data.Class == Class][['R', 'G', 'B']].mean().values for Class in data.loc[:,'C1':f'C{K}'].columns])
        i += 1
        print(f'Iteration: {i}\tDistance: {abs(new_centroids.mean() - old_centroids.mean())}')
    print(f"k-Means has ended with {i} iterations")
    return data, L


# In[49]:


k = 3

segmented_1 = {}
distances_1 = {}

# Perform k-means clustering with Euclidean distance on df1
segmented_1['euclidean'], distances_1['euclidean'] = kmeans(df.copy(), k)  # Avoid modifying original df

# Perform k-means clustering with Euclidean distance on df2 (assuming df2 is defined)
segmented_2 = {}
distances_2 = {}
segmented_2['euclidean'], distances_2['euclidean'] = kmeans(df2.copy(), k)  # Avoid modifying original df2


# In[50]:


fig, ax = plt.subplots(ncols=2, figsize=(16,9))
for key in distances_1.keys():
  ax[0].plot(distances_1[key], lw=2.5, label=key)
  ax[1].plot(distances_1[key], lw=2.5, label=key)

for i in range(2):
  ax[i].set_xlabel('Epoch')
  ax[i].set_ylabel('Centroid Distance')
  ax[i].grid(color='black', ls='--', lw=0.5)
  ax[i].legend()

ax[0].set_title('Img 2000-08-01',fontsize=20)
ax[1].set_title('Img 2023-08-01',fontsize=20)
plt.savefig('comparison.png')
plt.show()  


# In[51]:


d = {'C1':0, 'C2': 1, 'C3':2}
for key in segmented_1.keys():
  segmented_1[key].Class = segmented_1[key].Class.apply(lambda x: d[x])
  segmented_2[key].Class = segmented_2[key].Class.apply(lambda x: d[x])


# In[52]:


for key in segmented_1.keys():
  fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10,10))
  ax[0, 0].imshow(img)
  ax[0, 1].imshow(segmented_1[key].Class.values.reshape(460,390))
  ax[0, 0].set_title('MOD09GA RGB', fontsize=18)
  ax[0, 1].set_title(f'kMeans\n{key[0].upper()+key[1:]} Distance', fontsize=18)

  ax[1, 0].imshow(img2)
  ax[1, 1].imshow(segmented_2[key].Class.values.reshape(460,390))
  ax[1, 0].set_title('MOD09GA RGB', fontsize=18)
  ax[1, 1].set_title(f'kMeans\n{key[0].upper()+key[1:]} Distance', fontsize=18)

  for i in range(2):
    for j in range(2):
      ax[i, j].set_facecolor('black')
      ax[i, j].set_xticks([])
      ax[i, j].set_yticks([])

  plt.savefig(f'{key}.png')
  plt.tight_layout()
  plt.show()


# In[53]:


np.count_nonzero(segmented_1[key].Class.values == 1)


# In[54]:


IMG = segmented_1['euclidean'].Class.values.reshape(460,390).copy().astype(float)
mask = IMG!= 2
IMG[mask] = np.nan
plt.imshow(IMG)


# In[55]:


IMG = segmented_2['euclidean'].Class.values.reshape(460,390).copy().astype(float)
mask = IMG!= 2
IMG[mask] = np.nan
plt.imshow(IMG)


# In[57]:


for metric, Class in zip(['euclidean'], [2,1]):
  img1_water = np.count_nonzero(segmented_1[metric].Class.values == Class)*500*500*1e-6
  img2_water = np.count_nonzero(segmented_2[metric].Class.values == Class)*500*500*1e-6

  print(f'Distance: {metric}\tWater Area Before: {round(img1_water)}km\u00b2\tWater Area After: {round(img2_water)}km\u00b2\tChange: -{100-round(img2_water/img1_water*100)}%')


# In[ ]:




