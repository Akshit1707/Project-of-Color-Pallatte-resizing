import cv2
import numpy as np
from sklearn.cluster import KMeans

def get_color_palette(image_path, num_colors=5):
    #Read the image
    image=cv2.imread(image_path)
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #Flatten the image array
    pixels=image.reshape((-1, 3))

    #Apply K-means clustering
    kmeans=KMeans(n_clusters=num_colors)
    kmeans.fit(pixels)

    #Get the dominant colors
    dominant_colors = kmeans.cluster_centers_.astype(int)

    return dominant_colors


#Example usage
image_path= 'C:\WORK\VS code\Color Pallatte\istockphoto.jpg'
num_colors=5
palette= get_color_palette(image_path, num_colors)
print("Color Palatte:", palette)