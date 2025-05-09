# ComfyUI-Color_Transfer

Postprocessing nodes that implement color palette transfer in images. It replaces the dominant colors in an image with a target color palette.

## Prevention/troubleshooting of import error

To avoid import issues install python package using CLI ```pip install scikit-learn```.

## Installation

Clone repo to your custom_nodes folder
```git clone https://github.com/45uee/ComfyUI-Color_Transfer.git```

## Usage

1. Create a "Color Palette" node containing RGB values of your desired colors. Color must be defined in this format: [(Value, Value, Value), ...], for example [(30, 32, 30), (60, 61, 55), (105, 117, 101), (236, 223, 204)]
2. Create a "Palette Transfer" node, and connect your image and palette as input, that's all.
   
 -  You can specify color clustering and comparing distance method, by default MiniBatchKMeans faster but can be less accurate

## Example

![alt text](https://github.com/45uee/ComfyUI-Color_Transfer/blob/main/color_transfer_example.png)
