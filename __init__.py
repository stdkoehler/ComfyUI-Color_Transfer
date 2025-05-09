from .color_transfer import PaletteTransferNode, PalleteTransferClustering, PaletteTransferReinhard, PaletteSoftTransfer, PaletteRbfTransfer, PaletteOptimalTransportTransfer, ColorTransferReinhard, ColorPaletteNode


NODE_CLASS_MAPPINGS = {
    "PaletteTransfer": PaletteTransferNode,
    "PalleteTransferClustering": PalleteTransferClustering,
    "PaletteTransferReinhard": PaletteTransferReinhard,
    "PalletteSoftTransfer": PaletteSoftTransfer,
    "PaletteRbfTransfer": PaletteRbfTransfer,
    "PaletteOptimalTransportTransfer": PaletteOptimalTransportTransfer,
    "ColorTransferReinhard": ColorTransferReinhard,
    "ColorPalette": ColorPaletteNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PaletteTransfer": "Palette Transfer",
}
