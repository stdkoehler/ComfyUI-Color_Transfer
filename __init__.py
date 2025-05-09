from .color_transfer import PaletteTransferNode, PalleteTransferClustering, ColorPaletteNode


NODE_CLASS_MAPPINGS = {
    "PaletteTransfer": PaletteTransferNode,
    "PalleteTransferClustering": PalleteTransferClustering,
    "ColorPalette": ColorPaletteNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PaletteTransfer": "Palette Transfer",
}
