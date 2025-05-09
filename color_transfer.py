from itertools import combinations

import numpy as np
import torch
import ast
import cv2

from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.spatial import Delaunay

from comfy.comfy_types import IO, ComfyNodeABC

from .utils import (
    EuclideanDistance,
    ManhattanDistance,
    CosineSimilarity,
    HSVColorSimilarity,
    RGBWeightedDistance,
    RGBWeightedSimilarity,
    Blur
)

class PaletteExtension:
    @staticmethod
    def dense_palette(base_palette, points=5, iterations=2):
        """
        Interpolate N points between each pair of colors in the base_palette.
        Do the same for the new colors generated, for a given number of iterations to
        create a dense mesh of interpolated colors.
        """
        # add black and white to the base palette
        base_palette = set(base_palette)
        base_palette.add((0,0,0))
        base_palette.add((255,255,255))

        palette = np.array(list(base_palette), dtype=float)

        for _ in range(iterations):
            # build all combinations of two distinct indices
            idx_pairs = list(combinations(range(len(palette)), 2))

            # precompute interpolation fractions
            t = np.linspace(0, 1, points + 2)[1:-1]

            # for each pair, generate points intermediates
            new_colors = []
            for i, j in idx_pairs:
                c1, c2 = palette[i], palette[j]
                # broadcast interpolation in one go:
                inter = c1[None, :] + (c2 - c1)[None, :] * t[:, None]
                new_colors.append(inter)
            if new_colors:
                new_colors = np.vstack(new_colors)
                palette = np.vstack([palette, new_colors])

            # remove duplicates
            palette = np.unique(np.rint(palette).astype(int), axis=0).astype(float)

        # final unique, integer RGB list
        result = [tuple(rgb.astype(int)) for rgb in palette]
        return result

    @staticmethod
    def edge_based_palette(base_palette, points=5, iterations=2):
        """
        Use Delaunay triangulation to find edges between colors in the base_colors.
        Do the same for the new colors generated, for a given number of iterations to
        Create a dense mesh of interpolated colors.
        In contrast to dense_palette, this method uses the edges of the triangulation
        to generate new colors, which can lead to a more structured palette.
        """
        # add black and white to the base palette
        base_palette = set(base_palette)
        base_palette.add((0,0,0))
        base_palette.add((255,255,255))

        palette = np.array(list(base_palette), dtype=float)

        for _ in range(iterations):
            # Triangulate current palette to find adjacency edges
            if len(palette) >= palette.shape[1] + 1:
                tri = Delaunay(palette)
                edges = set()
                for simplex in tri.simplices:
                    # add all edges of the simplex
                    for i in range(len(simplex)):
                        for j in range(i+1, len(simplex)):
                            a, b = simplex[i], simplex[j]
                            edges.add(tuple(sorted((a, b))))
            else:
                # fallback to all pairs
                idxs = range(len(palette))
                edges = set(tuple(sorted((i,j))) for i in idxs for j in idxs if i<j)

            new_colors = []
            t = np.linspace(0, 1, points+2)[1:-1][:, None]  # shape (points,1)

            for i, j in edges:
                c1, c2 = palette[i], palette[j]
                inters = c1 + (c2 - c1) * t  # (points,3)
                new_colors.append(inters)

            if new_colors:
                new_stack = np.vstack(new_colors)
                palette = np.vstack([palette, new_stack])
                # remove duplicates
                palette = np.unique(np.rint(palette).astype(int), axis=0).astype(float)

        return [tuple(c.astype(int)) for c in palette]

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

class PalleteTransferClustering(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls):
        data_in = {
            "required": {
                "image": (IO.IMAGE,),
                "target_colors": ("COLOR_LIST",),
                "palette_extension_method": (["Dense", "Edge", "None"], {'default': 'None'}),
                "palette_extension_points": (IO.INT, {'min': 2, 'max': 20, 'step': 1, 'default': 5,}),
                "gaussian_blur": (IO.INT, {'default': 3, 'min': 0, 'max': 27, 'step': 1}),
                }
            }
        return data_in

    RETURN_TYPES = (IO.IMAGE,)
    FUNCTION = "color_transfer"
    CATEGORY = "Palette Transfer"

    def color_transfer(self, image, target_colors, palette_extension_method, palette_extension_points, gaussian_blur):
        if len(target_colors) == 0:
            return (image,)

        processedImages = []

        if palette_extension_method == "Dense":
            target_colors = PaletteExtension.dense_palette(target_colors, points=palette_extension_points)
        elif palette_extension_method == "Edge":
            target_colors = PaletteExtension.edge_based_palette(target_colors, points=palette_extension_points)



        # Initialize components
        converter = ColorSpaceConvert()
        clustering_engine = ColorClustering("Mini batch Kmeans")
        color_matcher = ColorMatcher("Euclidean")
        image_processor = ImagePostProcessor(gaussian_blur)

        for img_tensor in image:
            # Prepare image
            img = 255. * img_tensor.cpu().numpy()

            # Convert color space
            converted_img, converted_colors = converter.convert_to_target_space(img, target_colors, "RGB")

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
            processed = converter.convert_to_rgb(processed, "RGB")

            # Post-process
            processed = image_processor.process_image(processed)
            processed_tensor = torch.from_numpy(processed)[None,]

            processedImages.append(processed_tensor)

        output = torch.cat(processedImages, dim=0)
        return (output,)

class PaletteTransferNode(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls):
        data_in = {
            "required": {
                "image": (IO.IMAGE,),
                "target_colors": ("COLOR_LIST",),
                "color_space": (["RGB", "HSV", "LAB"], {'default': 'RGB'}),
                "cluster_method": (["Kmeans","Mini batch Kmeans"], {'default': 'Kmeans'}, ),
                "distance_method": (["Euclidean", "Manhattan", "Cosine Similarity", "HSV Distance", "RGB Weighted Distance", "RGB Weighted Similarity"], {'default': 'Euclidean'}, ),
                "gaussian_blur": (IO.INT, {'default': 3, 'min': 0, 'max': 27, 'step': 1}),
                }
            }
        return data_in

    RETURN_TYPES = (IO.IMAGE,)
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

        for img_tensor in image:
            # Prepare image
            img = 255. * img_tensor.cpu().numpy()

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


class ColorPaletteNode(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "color_palette": (IO.STRING, {'default': '[(30, 32, 30), (60, 61, 55), (105, 117, 101), (236, 223, 204)]', 'multiline': True}),
            },
        }


    RETURN_TYPES = ("COLOR_LIST", )
    RETURN_NAMES = ("Color palette", )
    FUNCTION = "color_list"
    CATEGORY = "Palette Transfer"

    def color_list(self, color_palette):
        return (ast.literal_eval(color_palette), )
