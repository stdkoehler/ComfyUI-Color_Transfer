import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
import torch
import ast
import cv2
from .utils import (
    EuclideanDistance,
    ManhattanDistance,
    CosineSimilarity,
    HSVColorSimilarity,
    RGBWeightedDistance,
    RGBWeightedSimilarity,
    Blur
)


class ColorSpaceConvert:
    @staticmethod
    def convert_to_target_space(image, target_colors, color_space):
        """Convert image and target colors to specified color space."""
        if color_space == "RGB":
            return image, target_colors
            
        conversion_map = {
            "HSV": (cv2.COLOR_RGB2HSV, cv2.COLOR_HSV2RGB),
            "LAB": (cv2.COLOR_RGB2LAB, cv2.COLOR_LAB2RGB)
        }
        
        forward_conversion, _ = conversion_map[color_space]
        
        converted_image = cv2.cvtColor(image, forward_conversion)
        
        target_colors_array = np.array(target_colors, dtype=np.uint8).reshape(-1, 1, 3)
        converted_colors = cv2.cvtColor(target_colors_array, forward_conversion)
        converted_colors = [tuple(color[0]) for color in converted_colors]
        
        return converted_image, converted_colors

    @staticmethod
    def convert_to_rgb(image, color_space):
        """Convert image back to RGB color space."""
        if color_space == "RGB":
            return image
            
        conversion_map = {
            "HSV": cv2.COLOR_HSV2RGB,
            "LAB": cv2.COLOR_LAB2RGB
        }
        
        return cv2.cvtColor(image, conversion_map[color_space])


class ColorClustering:
    def __init__(self, cluster_method):
        self.clustering_methods = {
            "Kmeans": KMeans,
            "Mini batch Kmeans": MiniBatchKMeans
        }
        self.method = self.clustering_methods[cluster_method]

    def cluster_colors(self, image, k):
        """Perform color clustering on the image."""
        img_array = image.reshape((-1, 3))
        clustering_model = self.method(n_clusters=k, n_init='auto')
        clustering_model.fit(img_array)
        
        return {
            'image': image,
            'main_colors': clustering_model.cluster_centers_.astype(int),
            'model': clustering_model
        }


class ColorMatcher:
    def __init__(self, distance_method):
        self.distance_methods = {
            "Euclidean": EuclideanDistance,
            "Manhattan": ManhattanDistance,
            "Cosine Similarity": CosineSimilarity,
            "HSV Distance": HSVColorSimilarity,
            "RGB Weighted Distance": RGBWeightedDistance,
            "RGB Weighted Similarity": RGBWeightedSimilarity
        }
        self.distance_func = self.distance_methods[distance_method]

    def match_colors(self, detected_colors, target_colors, clustering_model, image_shape):
        """Match detected colors with target colors using the specified distance method."""
        closest_colors = []
        
        for color in detected_colors:
            distances = self.distance_func(color, target_colors)
            closest_color = target_colors[np.argmin(distances)]
            closest_colors.append(closest_color)
            
        closest_colors = np.array(closest_colors)
        return closest_colors[clustering_model.labels_].reshape(image_shape)


class ImagePostProcessor:
    def __init__(self, gaussian_blur=0):
        self.gaussian_blur = gaussian_blur

    def process_image(self, image):
        """Apply post-processing to the image."""
        processed = np.array(image).astype(np.float32)
        
        if self.gaussian_blur:
            processed = Blur(processed, self.gaussian_blur)
            
        return processed / 255.0
    

class PaletteTransferNode:
    @classmethod
    def INPUT_TYPES(cls):
        data_in = {
            "required": {
                "image": ("IMAGE",),
                "target_colors": ("COLOR_LIST",),
                "color_space": (["RGB", "HSV", "LAB"], {'default': 'RGB'}),
                "cluster_method": (["Kmeans","Mini batch Kmeans"], {'default': 'Kmeans'}, ),
                "distance_method": (["Euclidean", "Manhattan", "Cosine Similarity", "HSV Distance", "RGB Weighted Distance", "RGB Weighted Similarity"], {'default': 'Euclidean'}, ),
                "gaussian_blur": ("INT", {'default': 3, 'min': 0, 'max': 27, 'step': 1}),
                }
            }
        return data_in

    CATEGORY = "Color Transfer"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "color_transfer"
    CATEGORY = "Palette Transfer"


    def color_transfer(self, image, target_colors, color_space, cluster_method, distance_method, gaussian_blur):
        if len(target_colors) == 0:
            return (image,)
        
        processedImages = []
        
        # Initialize components
        converter = ColorSpaceConvert()
        clustering_engine = ColorClustering(cluster_method)
        color_matcher = ColorMatcher(distance_method)
        image_processor = ImagePostProcessor(gaussian_blur)

        for img in image:
            # Prepare image
            img = 255. * img.cpu().numpy()
            
            # Convert color space
            converted_img, converted_colors = converter.convert_to_target_space(img, target_colors, color_space)
            
            # Perform clustering
            clustering_result = clustering_engine.cluster_colors(converted_img, len(target_colors))
            
            # Match colors
            processed = color_matcher.match_colors(
                clustering_result['main_colors'],
                converted_colors,
                clustering_result['model'],
                converted_img.shape
            )
            
            # Convert back to RGB
            processed = converter.convert_to_rgb(processed, color_space)
            
            # Post-process
            processed = image_processor.process_image(processed)
            processed_tensor = torch.from_numpy(processed)[None,]
            
            processedImages.append(processed_tensor)
        
        output = torch.cat(processedImages, dim=0)
        return (output,)
    

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
