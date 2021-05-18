import json
import os

import cv2
import pandas as pd

import datasets
import torch

from detectron2.data.detection_utils import read_image
from detectron2.data.transforms import ResizeTransform, TransformList

logger = datasets.logging.get_logger(__name__)

_CITATION = """\
@DmitryKoryakov
"""

_DESCRIPTION = """\
nani
"""


class FRConfig(datasets.BuilderConfig):
    """BuilderConfig for FR"""

    def __init__(self, **kwargs):
        """BuilderConfig for FR.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(FRConfig, self).__init__(**kwargs)


class FR(datasets.GeneratorBasedBuilder):
    """Conll2003 dataset_."""

    BUILDER_CONFIGS = [
        FRConfig(name="FR", version=datasets.Version("1.0.0"), description="FR"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=['O',
                                   'PRODUCT',
                                   'QTY',
                                   'TOTAL',
                                   'UNIT',
                                   'COMPANY',
                                   'REFERENCE_ID',
                                   'DATE',
                                   'TAX',
                                   'ALL_TAX',
                                   'ALL_TOTAL',
                                   'SUBTOTAL',
                                   'UNITTYPE',
                                   ]
                            # names=['O',
                            #        'B-PRODUCT',
                            #        'I-PRODUCT',
                            #        'B-QTY',
                            #        'I-QTY',
                            #        'B-TOTAL',
                            #        'I-TOTAL',
                            #        'B-UNIT',
                            #        'I-UNIT',
                            #        'B-COMPANY',
                            #        'I-COMPANY',
                            #        'B-REFERENCE_ID',
                            #        'I-REFERENCE_ID',
                            #        'B-DATE',
                            #        'I-DATE',
                            #        'B-TAX',
                            #        'I-TAX',
                            #        'B-ALL_TAX',
                            #        'I-ALL_TAX',
                            #        'B-ALL_TOTAL',
                            #        'I-ALL_TOTAL',
                            #        'B-SUBTOTAL',
                            #        'I-SUBTOTAL',
                            #        'B-UNITTYPE',
                            #        'I-UNITTYPE',
                            #        ]

                        )
                    ),
                    "image": datasets.Array3D(shape=(3, 224, 224), dtype="uint8"),
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/KoryakovDmitry/ltlmv2.git",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # downloaded_file = dl_manager.download_and_extract("https://guillaumejaume.github.io/FUNSD/dataset.zip")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": os.path.join("/Volumes/ssd/Initflow/ltlmv2", "dataset/training_data/")}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": os.path.join("/Volumes/ssd/Initflow/ltlmv2", "dataset/testing_data/")}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": os.path.join("/Volumes/ssd/Initflow/ltlmv2", "dataset/val_data/")}
            ),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        ann_dir = os.path.join(filepath, "annotations")
        img_dir = os.path.join(filepath, "images")
        for guid, file in enumerate(sorted(os.listdir(ann_dir))):
            bboxes = []
            tokens = []
            ner_tags = []

            file_path = os.path.join(ann_dir, file)

            data = pd.read_csv(file_path)

            image_path = os.path.join(img_dir, file)
            image_path = image_path.replace("csv", "jpg")
            image, size = load_image(image_path)

            data = data[data.label.apply(lambda x: False if x in ("date_within", "company_within", "reference_id_within", "logo") else True)].reset_index(drop=True)

            for idx, row in data.iterrows():
                x, y, w, h = cv2.boundingRect(row[:8].to_numpy().reshape(-1, 2).astype(int))
                bbox = [x, y, x+w, y+h]
                bboxes.append(normalize_bbox(bbox, size))

                tokens.append(row.text)
                label = row.label

                if label == "negative":
                    ner_tags.append("O")
                else:
                    ner_tags.append(label.upper())

            yield guid, {"id": str(guid), "tokens": tokens, "bboxes": bboxes, "ner_tags": ner_tags, "image": image}

    def _generate_examples_old(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        ann_dir = os.path.join(filepath, "annotations")
        img_dir = os.path.join(filepath, "images")
        for guid, file in enumerate(sorted(os.listdir(ann_dir))):
            tokens = []
            bboxes = []
            ner_tags = []

            file_path = os.path.join(ann_dir, file)
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)
            image_path = os.path.join(img_dir, file)
            image_path = image_path.replace("json", "png")
            image, size = load_image(image_path)
            for item in data["form"]:
                words, label = item["words"], item["label"]
                words = [w for w in words if w["text"].strip() != ""]
                if len(words) == 0:
                    continue
                if label == "other":
                    for w in words:
                        tokens.append(w["text"])
                        ner_tags.append("O")
                        bboxes.append(normalize_bbox(w["box"], size))
                else:
                    tokens.append(words[0]["text"])
                    ner_tags.append("B-" + label.upper())
                    bboxes.append(normalize_bbox(words[0]["box"], size))
                    for w in words[1:]:
                        tokens.append(w["text"])
                        ner_tags.append("I-" + label.upper())
                        bboxes.append(normalize_bbox(w["box"], size))

            yield guid, {"id": str(guid), "tokens": tokens, "bboxes": bboxes, "ner_tags": ner_tags, "image": image}


def normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]


def load_image(image_path):
    image = read_image(image_path, format="BGR")
    h = image.shape[0]
    w = image.shape[1]
    img_trans = TransformList([ResizeTransform(h=h, w=w, new_h=224, new_w=224)])
    image = torch.tensor(img_trans.apply_image(image).copy()).permute(2, 0, 1)  # copy to make it writeable
    return image, (w, h)
