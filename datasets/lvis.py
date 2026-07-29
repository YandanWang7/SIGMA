from pathlib import Path
import os
import random
import copy
from .lvis_v1_categories import LVIS_CATEGORIES as LVIS_V1_CATEGORIES
from .coco import ConvertCocoPolysToMask
from . import transforms as T
import torch
from PIL import Image


class _SimpleLVIS:
    def __init__(self, json_file):
        import json

        with open(json_file, "r", encoding="utf-8") as handle:
            self.dataset = json.load(handle)
        self.imgs = {img["id"]: img for img in self.dataset.get("images", [])}
        self.img_ann_map = {img_id: [] for img_id in self.imgs.keys()}
        for ann in self.dataset.get("annotations", []):
            self.img_ann_map.setdefault(ann["image_id"], []).append(ann)

    def load_imgs(self, img_ids):
        return [self.imgs[img_id] for img_id in img_ids]


def load_lvis_json(json_file, image_root, dataset_name=None):
    """
    Load a json file in LVIS's annotation format.

    Args:
        json_file (str): full path to the LVIS json annotation file.
        image_root (str): the directory where the images in this json file exists.
        dataset_name (str): the name of the dataset (e.g., "lvis_v0.5_train").
            If provided, this function will put "thing_classes" into the metadata
            associated with this dataset.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    try:
        from lvis import LVIS
        lvis_api = LVIS(json_file)
    except ImportError:
        lvis_api = _SimpleLVIS(json_file)

    meta = get_lvis_instances_meta(dataset_name)
    img_ids = sorted(lvis_api.imgs.keys())

    imgs = lvis_api.load_imgs(img_ids)
    anns = [lvis_api.img_ann_map[img_id] for img_id in img_ids]

    ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
    assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique".format(
        json_file
    )

    imgs_anns = list(zip(imgs, anns))

    def get_file_name(img_root, img_dict):
        if 'file_name' in img_dict:
            file_name = img_dict['file_name']
            split_folder = ''
        else:
            split_folder, file_name = img_dict["coco_url"].split("/")[-2:]
        return os.path.join(os.path.join(img_root, split_folder), file_name)

    dataset_dicts = []

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = get_file_name(image_root, img_dict)
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        record["not_exhaustive_category_ids"] = img_dict.get("not_exhaustive_category_ids", [])
        record["neg_category_ids"] = img_dict.get("neg_category_ids", [])
        if 'pos_category_ids' in img_dict:
            record['pos_category_ids'] = img_dict.get("pos_category_ids", [])  # for custom imagenet-lvis
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            assert anno["image_id"] == image_id
            obj = {"bbox": anno["bbox"]}  # XYHW mode

            # FIXME: stop converting to 0-nd
            obj["category_id"] = anno["category_id"]

            if 'segmentation' in anno:
                segm = anno["segmentation"]
                valid_segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                assert len(segm) == len(
                    valid_segm
                ), "Annotation contains an invalid polygon with < 3 points"
                assert len(segm) > 0
                obj["segmentation"] = segm
            obj['area'] = anno['area']
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts, lvis_api


def get_lvis_instances_meta(dataset_name):
    if 'v1' in dataset_name:
        return _get_lvis_v1_meta()
    raise ValueError("No built-in metadata for dataset {}".format(dataset_name))


def _get_lvis_v1_meta():
    assert len(LVIS_V1_CATEGORIES) == 1203
    cat_ids = [k["id"] for k in LVIS_V1_CATEGORIES]
    assert min(cat_ids) == 1 and max(cat_ids) == len(
        cat_ids
    ), "Category ids are not in [1, #categories], as expected"
    # Ensure that the category list is sorted by id
    lvis_categories = sorted(LVIS_V1_CATEGORIES, key=lambda x: x["id"])
    thing_classes = [k["synonyms"][0] for k in lvis_categories]
    meta = {"thing_classes": thing_classes}
    return meta


class LvisDetection():
    def __init__(self, json_file, image_root, dataset_name, transforms, return_masks, is_extra=False):
        self.data, self.lvis = load_lvis_json(json_file, image_root, dataset_name)
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self._transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            data = self.data[idx]
        except:
            idx = random.randint(0, len(self))
            data = self.data[idx]

        data = copy.deepcopy(data)
        anno = data.pop('annotations', [])
        img = Image.open(data['file_name']).convert('RGB')
        image_id = data['image_id']
        target = {'image_id': image_id, 'annotations': anno}

        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target

def build(image_set, args):
    root = args.coco_path
    # assert root.exists(), f'provided COCO path {root} does not exist'
    # this arg is for using partial annotation in lvis
    train_path = 'annotations/lvis_v1_train.json'

    if args.dataset_file == 'lvis':
        PATHS = {
            "train": (root, os.path.join(root, train_path)),
            "val": (root, os.path.join(root, 'annotations/lvis_v1_val.json')),
        }

    try:
        strong_aug = args.strong_aug
    except:
        strong_aug = False

    from .coco import make_coco_transforms
    dataset = LvisDetection(PATHS[image_set][1], root, 'lvis_v1',
                            transforms=make_coco_transforms(image_set, fix_size=args.fix_size, strong_aug=strong_aug,
                                                            args=args),
                            return_masks=args.masks,
                            )
    return dataset





