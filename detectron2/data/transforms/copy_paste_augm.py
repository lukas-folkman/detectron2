import numpy as np
import sys
import os
import torch
from detectron2.structures import BoxMode, Boxes
import cv2
import skimage
from fvcore.transforms.transform import Transform
from .augmentation import Augmentation
import json
from random import sample
from cv_tools import merge_multi_segment, filter_annotations_with_images, pairwise_ioa


DEBUG = False


def unscaled_prob(cat_id, cat_fractions=None, t=1, strength='sqrt_balance'):
    if cat_fractions is None:
        cat_fractions = [0.7242705762810278, 0.21686746987951808, 0.0588619538394542]

    if strength == 'full_balance':
        return cat_fractions[0] / cat_fractions[int(cat_id)]
    elif strength == 'closed_balance':
        return min(5, cat_fractions[0] / cat_fractions[int(cat_id)])
    elif strength == 'sqrt_balance':
        return max(1, np.sqrt(t / cat_fractions[int(cat_id)]))
    else:
        raise ValueError


def load_image(file_path):
    image = cv2.imread(file_path)
    if image is None:
        raise FileNotFoundError(f"No image found at {file_path}")
    return image


def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def format_segments(segments):
    # LF - if there are disconnected segments, merge them
    if len(segments) > 1:
        segments = [np.concatenate(merge_multi_segment(segments), axis=0).astype(np.int32).reshape(-1, 2)]
    else:
        assert len(segments) == 1
        segments = [np.array(segments[0], dtype=np.int32).reshape(-1, 2)]
    return segments


def process_dataset(json_fn, img_dir):
    data = load_json(json_fn)

    if DEBUG:
        data['images'] = data['images'][:100]
        data['annotations'] = filter_annotations_with_images(data)

    image_data = {image['id']: load_image(os.path.join(img_dir, image['file_name'])) for image in data['images']}
    annotations_data = [dict(
        img_id=ann['image_id'],
        bbox=ann['bbox'],
        segmentation=format_segments(ann['segmentation']),
        category_id=ann['category_id'] - 1,
        bbox_mode=BoxMode.XYWH_ABS,
        iscrowd=0,
    ) for ann in data['annotations']]

    assert set([ann['img_id'] for ann in annotations_data]).issubset(list(image_data.keys()))

    return annotations_data, image_data


class CopyPasteAugmentation(Augmentation):

    def __init__(self, cp_type, n_objects, max_ioa, json_fn, img_dir, seed):
        assert n_objects > 0
        assert cp_type in ['random', 'sqrt_balance', 'closed_balance', 'full_balance']
        super(CopyPasteAugmentation, self).__init__()
        # load the dataset to copy from
        self.cp_type = cp_type
        self.max_ioa = max_ioa
        self.n_objects = n_objects
        self.json_fn = json_fn
        self.img_dir = img_dir
        self.random_state = np.random.RandomState(seed=seed)
        self.resampling_p = None
        self.verbose = True

        if self.verbose:
            print(f'CPA: enabled, num_fish: {self.n_objects}, cp_type: {self.cp_type}, max_ioa: {self.max_ioa}')

        self.annotations_data, self.image_data = process_dataset(self.json_fn, self.img_dir)  # LF

        if self.cp_type != 'random':
            # https://arxiv.org/pdf/2012.07177
            # https://arxiv.org/pdf/1908.03195
            self.resampling_p = np.asarray([
                unscaled_prob(ann['category_id'], strength=self.cp_type) for ann in self.annotations_data
            ])
            self.resampling_p /= np.sum(self.resampling_p)

            if self.verbose:
                CAT_ID_COUNTS = [9290, 2814, 720]
                for cat_id in range(len(CAT_ID_COUNTS)):
                    cat_id_mask = [int(ann['category_id']) == cat_id for ann in self.annotations_data]
                    if self.verbose:
                        print('CPA: category =', cat_id, 'P =',
                              self.resampling_p[np.asarray(cat_id_mask)][0] * CAT_ID_COUNTS[cat_id])


    def get_transform(self, image, boxes):
        cpa_annotations = []
        cpa_collage = np.zeros(image.shape, dtype=image.dtype)
        cpa_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        pasted_count = 0
        for ann_id in self.random_state.choice(
                range(len(self.annotations_data)), size=len(self.annotations_data), replace=False,
                p=self.resampling_p
        ):
            ann = self.annotations_data[ann_id]
            assert len(ann['bbox']) == 4 and isinstance(ann['bbox'][0], (float, int)), ann['bbox']
            assert len(ann['segmentation']) == 1 and ann['segmentation'][0].shape[1] == 2 and \
                   ann['segmentation'][0].dtype == np.int32, ann['segmentation']
            if len(boxes) == 0 or np.all(pairwise_ioa(np.asarray([ann['bbox']]), boxes) < self.max_ioa):
                cpa_annotations.append(
                    dict(bbox=ann['bbox'], bbox_mode=ann['bbox_mode'],
                         category_id=ann['category_id'], iscrowd=ann['iscrowd'])
                )

                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, ann['segmentation'], -1, (255, 255, 255), cv2.FILLED)
                cpa_collage[np.where(mask)] = self.image_data[ann['img_id']][np.where(mask)]
                cpa_mask[np.where(mask)] = 1
                pasted_count += 1
                if pasted_count >= self.n_objects:
                    break

        return CopyPasteTransform(cpa_annotations, cpa_collage, cpa_mask)

class CopyPasteTransform(Transform):
    def __init__(self, cpa_annotations, cpa_collage, cpa_mask):
        super(CopyPasteTransform, self).__init__()
        self.cpa_annotations = cpa_annotations
        self.cpa_collage = cpa_collage
        self.cpa_mask = cpa_mask

    def apply_image(self, img: np.ndarray):
        img_copy = img.copy()
        img_copy[np.where(self.cpa_mask)] = self.cpa_collage[np.where(self.cpa_mask)]
        return img_copy

    def apply_annotations(self, annotations: list) -> list:
        return np.concatenate((annotations, self.cpa_annotations)).tolist()

    def apply_coords(self, coords: np.ndarray):
        return coords
