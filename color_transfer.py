import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
import torch
import ast
import cv2
from .utils import EuclideanDistance, ManhattanDistance




def ColorClustering(image, k, cluster_method):
    img_array = image.reshape((image.shape[0] * image.shape[1], 3))

    cluster_methods = {
    "Kmeans": KMeans,
    "Mini batch Kmeans": MiniBatchKMeans
    }

    clustering_model = cluster_methods.get(cluster_method)(n_clusters=k, n_init='auto')

    clustering_model.fit(img_array)
    main_colors = clustering_model.cluster_centers_
    return image, main_colors.astype(int), clustering_model


def SwitchColors(image, detected_colors, target_colors, clustering_model, distance_method):
    closest_colors = []

    distance_methods = {
    "Euclidean": EuclideanDistance,
    "Manhattan": ManhattanDistance
    }

    distance_method = distance_methods.get(distance_method)

    for color in detected_colors:
        distances = distance_method(color, target_colors)
        closest_color = target_colors[np.argmin(distances)]
        closest_colors.append(closest_color)

    closest_colors = np.array(closest_colors)

    image = closest_colors[clustering_model.labels_].reshape(image.shape)
    
    return image


class PaletteTransferNode:
    @classmethod
    def INPUT_TYPES(cls):
        data_in = {
            "required": {
                "image": ("IMAGE",),
                "target_colors": ("COLOR_LIST",),
                "color_space": (["RGB", "HSV", "LAB"], {'default': 'RGB'}),
                "cluster_method": (["Kmeans","Mini batch Kmeans"], {'default': 'Kmeans'}, ),
                "distance_method": (["Euclidean", "Manhattan"], {'default': 'Euclidean'}, )
                }
            }
        return data_in


    CATEGORY = "Color Transfer"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "color_transfer"
    CATEGORY = "Palette Transfer"


    def color_transfer(self, image, target_colors, color_space, cluster_method, distance_method):

        if len(target_colors) == 0:
            return (image,)
        
        processedImages = []

        for image in image:
            img = 255. * image.cpu().numpy()

            if color_space == "HSV":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

                target_colors = np.array(target_colors, dtype=np.uint8).reshape(-1, 1, 3)
                target_colors = cv2.cvtColor(target_colors, cv2.COLOR_RGB2HSV)
                target_colors = [tuple(hsv[0]) for hsv in target_colors]


            clustered_img, detected_colors, clustering_model = ColorClustering(img, len(target_colors), cluster_method)
            processed = SwitchColors(clustered_img, detected_colors, target_colors, clustering_model, distance_method)

            if color_space == "HSV":
                processed = cv2.cvtColor(processed, cv2.COLOR_HSV2RGB)
            
            processed = np.array(processed).astype(np.float32) / 255.0
            processedImage = torch.from_numpy(processed)[None,]

            processedImages.append(processedImage)
        
        output = torch.cat(processedImages, dim=0)

        return (output, )
    

class ColorPaletteNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "color_palette": ("STRING", {'default': '[(30, 32, 30), (60, 61, 55), (105, 117, 101), (236, 223, 204)]', 'multiline': True}),
            },
        }

    CATEGORY = "Color Transfer"
    RETURN_TYPES = ("COLOR_LIST", )
    RETURN_NAMES = ("Color palette", )
    FUNCTION = "color_list"

    def color_list(self, color_palette):
        return (ast.literal_eval(color_palette), )
