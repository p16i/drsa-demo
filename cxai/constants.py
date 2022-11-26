from frozendict import frozendict

NUMBER_SPATIAL_LOCATIONS_SELECTED = 20

ARCH_MAIN_LAYERS = {
    "torchvision-vgg16-imagenet": "conv4_3",
    "netdissect-vgg16-imagenet": "conv4_3",
    "dm_nfnet_f0": "stage2",
}

NUMBER_SUBSPACES = 4

DRSA_SOFTMIN_MAIN_ORD = 2

# Ref: https://github.com/CSAILVision/NetDissect/blob/release1/script/rundissect.sh#L165
# These values are raw pixel values, e.g., [0, 255]
NETDISSECT_BGR_MEAN_RAW = (109.5388, 118.6897, 124.6901)

NETDISSECT_BGR_MEAN = tuple(map(lambda v: v / 255.0, NETDISSECT_BGR_MEAN_RAW))
NETDISSECT_RGB_MEAN = NETDISSECT_BGR_MEAN[::-1]

# Here, it seems that netdissect caffee models are trained on raw Pixel values
# So, we use these values to cancel out the normalize done by T.ToTensor()
NETDISSECT_RGB_STD = (1 / 255.0, 1 / 255.0, 1 / 255.0)

# Ref: https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L92
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

INPUT_SHAPE = frozendict(
    {
        "torchvision-vgg16-imagenet": (3, 224, 224),
        "netdissect-vgg16-imagenet": (3, 224, 224),
        "netdissect-vgg16-places365": (3, 224, 224),
        # Ref: https://github.com/rwightman/pytorch-image-models/blob/ee40b582bb67cbcb385b112bf519102c55d3d55a/timm/models/nfnet.py#L50
        "dm_nfnet_f0": (3, 192, 192),
        "dm_nfnet_f1": (3, 224, 224),
    }
)

PIXELFLIPPING_BASELINE_VALUE = 0


def get_arch_layer_dimensions(arch: str, layer: str) -> int:
    if arch in ["dm_nfnet_f0", "dm_nfnet_f1"]:
        if layer in ["stage2", "stage3"]:
            return 1536
        elif layer == "stage1":
            return 512
        elif layer == "stage0":
            return 256
        elif layer == "stem":
            return 128
        else:
            raise ValueError()
    elif arch in [
        "torchvision-vgg16-imagenet",
        "netdissect-vgg16-imagenet",
    ]:
        if layer in ["conv4_3", "conv5_3"]:
            return 512
        elif layer == "conv3_3":
            return 256
    else:
        raise ValueError(f"Not found layer dimension for `{arch}-{layer}`")


SHAPLEY_VALUE_SAMPLING_BATCH = 20
SHAPLEY_VALUE_SAMPLING_INPUT_PATCH_SIZE = 16
