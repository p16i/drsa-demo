import pandas as pd


from frozendict import frozendict

from cxai import constants


# todo: perhaps, this module should be in cxai/data/imagenet
df_label_mapping = pd.read_csv(
    constants.PACKAGE_DIR / "config" / "imagenet-label-mapping.csv"
)

# Taken from https://github.com/lightly-ai/lightly/blob/master/lightly/data/collate.py#L17
statistics = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


def get_index_from_imagenet_id(imagenet_id):
    row = df_label_mapping[df_label_mapping["imagenet-id"] == imagenet_id]

    assert len(row) == 1, f"we should have only one row {row}"

    return int(row.index[0])


def get_desc_from_imagenet_id(imagenet_id):
    row = df_label_mapping[df_label_mapping["imagenet-id"] == imagenet_id]

    return row.desc.values[0]


def get_desc_from_label_id(ix: int):
    nsid = get_imagenet_id_from_ix(ix)

    return get_desc_from_imagenet_id(nsid)


def get_all_imagenet_ids():
    return df_label_mapping["imagenet-id"].values


def get_imagenet_id_from_ix(ix: int) -> str:
    return df_label_mapping[df_label_mapping.index == ix]["imagenet-id"].values[0]


imgclasses = dict(zip(df_label_mapping.index.values, df_label_mapping.desc.values))

ix_to_classname = frozendict(imgclasses)
classname_to_ix = frozendict(
    dict(zip(ix_to_classname.values(), ix_to_classname.keys()))
)
