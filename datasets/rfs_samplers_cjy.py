import itertools
import math
from collections import defaultdict
from typing import Optional

import torch
from torch.utils.data.sampler import Sampler

from util.misc import all_gather, shared_random_seed
from util import misc


class RepeatFactorTrainingSampler(Sampler):
    """
    Similar to TrainingSampler, but a sample may appear more times than others
    according to its repeat factor. This is suitable for long-tailed detection
    datasets such as LVIS.
    """

    def __init__(self, repeat_factors, *, shuffle=True, seed=None):
        """
        Args:
            repeat_factors (Tensor): a float vector. The i-th element is the
                repeat factor for image i. When all elements are 1, this sampler
                degenerates to ordinary shuffled sampling.
            shuffle (bool): whether to shuffle the repeated indices.
            seed (int): initial seed. It must be shared across distributed ranks.
                If None, a shared random seed is generated.
        """
        self._shuffle = shuffle
        self._epoch = 0
        if seed is None:
            seed = shared_random_seed()
        self._seed = int(seed)

        self._rank = misc.get_rank()
        self._world_size = misc.get_world_size()

        repeat_factors = torch.as_tensor(repeat_factors, dtype=torch.float32)
        assert repeat_factors.dim() == 1, "repeat_factors must be a 1-D tensor."
        assert torch.all(repeat_factors >= 1.0), "repeat_factors must be >= 1."

        # Split into integer and fractional parts. The fractional part is handled
        # by stochastic rounding in each epoch, following the standard RFS logic.
        self._int_part = torch.trunc(repeat_factors)
        self._frac_part = repeat_factors - self._int_part

        # Kept for compatibility with the original code. The exact repeated
        # epoch size may vary slightly due to stochastic rounding.
        # sampler 长度是“先按原始长度报给训练框架，迭代后再变成重复长度”。如果训练框架依赖 len(dataloader) 来设置学习率调度或训练步数，就可能导致实验不够干净。
        # self.num_samples = int(math.ceil(len(self._int_part) * 1.0 / self._world_size))
        # self.total_size = self.num_samples * self._world_size
        # self.indices = []

        # wyd
        self.num_samples = int(math.ceil(float(repeat_factors.sum().item()) / self._world_size))
        self.total_size = self.num_samples * self._world_size
        self.indices = []

    def __len__(self) -> int:
        # The true length is known after __iter__ builds the repeated index list.
        # Before the first iteration, return the original compatibility estimate.
        # return len(self.indices) if len(self.indices) > 0 else self.num_samples
        # wyd
        return self.num_samples

    def set_epoch(self, epoch):
        self._epoch = epoch

    @staticmethod
    def repeat_factors_from_category_frequency(dataset_dicts, repeat_thresh):
        """
        Standard RFS.

        For each category c, compute image frequency f_c as the fraction of
        images containing at least one instance of c. Then compute
            r_c = max(1, sqrt(t / f_c)).
        For each image i, compute
            r_i = max_{c in i} r_c.
        """
        if repeat_thresh <= 0:
            return torch.ones(len(dataset_dicts), dtype=torch.float32)

        category_img_count = defaultdict(int)
        for dataset_dict in dataset_dicts:
            annotations = dataset_dict.get("annotations", [])
            cat_ids = {ann["category_id"] for ann in annotations if "category_id" in ann}
            for cat_id in cat_ids:
                category_img_count[cat_id] += 1

        num_images = len(dataset_dicts)
        if num_images == 0:
            return torch.empty(0, dtype=torch.float32)

        category_rep = {}
        for cat_id, img_count in category_img_count.items():
            img_freq = img_count / num_images
            category_rep[cat_id] = max(1.0, math.sqrt(repeat_thresh / img_freq))

        rep_factors = []
        for dataset_dict in dataset_dicts:
            annotations = dataset_dict.get("annotations", [])
            cat_ids = {ann["category_id"] for ann in annotations if "category_id" in ann}
            rep_factor = max((category_rep[cat_id] for cat_id in cat_ids), default=1.0)
            rep_factors.append(rep_factor)

        return torch.tensor(rep_factors, dtype=torch.float32)

    @staticmethod
    def repeat_factors_from_category_and_instance_frequency(
        dataset_dicts,
        repeat_thresh,
        mean_type="geometric",
    ):
        """
        IRFS: Instance-Aware Repeat Factor Sampling.

        This implementation follows the IRFS paper's default setting:
        geometric mean + t = repeat_thresh, usually 1e-3 on LVIS.

        For each category c:
            f_i,c = (# images containing c) / (# images)
            f_b,c = (# bbox instances of c) / (# bbox instances)

        With geometric mean:
            f_c = sqrt(f_i,c * f_b,c)
            r_c = max(1, sqrt(t / f_c))
                = max(1, sqrt(t / sqrt(f_i,c * f_b,c)))

        For each image i:
            r_i = max_{c in i} r_c

        Args:
            dataset_dicts (list[dict]): Detectron2-style dataset dicts. Each
                dict should contain an "annotations" list, and each annotation
                should contain "category_id".
            repeat_thresh (float): repeat threshold t. Use 1e-3 to reproduce the
                paper's default IRFS setting.
            mean_type (str): how to combine image frequency and instance
                frequency. The paper's default is "geometric". Other options are
                provided only for ablation consistency with the paper.
                Supported: "geometric", "harmonic", "arithmetic",
                "quadratic", "instance_only".

        Returns:
            torch.Tensor: per-image repeat factors.
        """
        if repeat_thresh <= 0:
            return torch.ones(len(dataset_dicts), dtype=torch.float32)

        num_images = len(dataset_dicts)
        if num_images == 0:
            return torch.empty(0, dtype=torch.float32)

        category_img_count = defaultdict(int)
        category_inst_count = defaultdict(int)
        total_instances = 0

        for dataset_dict in dataset_dicts:
            annotations = dataset_dict.get("annotations", [])
            image_cat_ids = set()

            for ann in annotations:
                if "category_id" not in ann:
                    continue
                cat_id = ann["category_id"]
                image_cat_ids.add(cat_id)
                category_inst_count[cat_id] += 1
                total_instances += 1

            # Image count is counted once per image per category, not per box.
            for cat_id in image_cat_ids:
                category_img_count[cat_id] += 1

        if total_instances == 0:
            return torch.ones(num_images, dtype=torch.float32)

        mean_type = mean_type.lower()
        valid_mean_types = {
            "geometric", "harmonic", "arithmetic", "quadratic", "instance_only"
        }
        if mean_type not in valid_mean_types:
            raise ValueError(
                f"Unsupported mean_type={mean_type!r}. "
                f"Expected one of {sorted(valid_mean_types)}."
            )

        category_rep = {}
        for cat_id, img_count in category_img_count.items():
            img_freq = img_count / num_images
            inst_freq = category_inst_count[cat_id] / total_instances

            # Both frequencies are strictly positive for categories observed in
            # category_img_count. Keep the guard for numerical safety.
            if img_freq <= 0 or inst_freq <= 0:
                continue

            if mean_type == "geometric":
                category_freq = math.sqrt(img_freq * inst_freq)
            elif mean_type == "harmonic":
                category_freq = 2.0 * img_freq * inst_freq / (img_freq + inst_freq)
            elif mean_type == "arithmetic":
                category_freq = 0.5 * (img_freq + inst_freq)
            elif mean_type == "quadratic":
                category_freq = math.sqrt(0.5 * (img_freq ** 2 + inst_freq ** 2))
            else:  # mean_type == "instance_only"
                category_freq = inst_freq

            category_rep[cat_id] = max(1.0, math.sqrt(repeat_thresh / category_freq))

        rep_factors = []
        for dataset_dict in dataset_dicts:
            annotations = dataset_dict.get("annotations", [])
            cat_ids = {ann["category_id"] for ann in annotations if "category_id" in ann}
            rep_factor = max((category_rep[cat_id] for cat_id in cat_ids), default=1.0)
            rep_factors.append(rep_factor)

        return torch.tensor(rep_factors, dtype=torch.float32)

    # Short alias. This is useful if you prefer calling IRFS explicitly in main.py.
    repeat_factors_from_irfs = repeat_factors_from_category_and_instance_frequency

    def _get_epoch_indices(self, generator):
        """
        Create a repeated index list for one epoch using stochastic rounding.
        """
        rands = torch.rand(len(self._frac_part), generator=generator)
        rep_factors = self._int_part + (rands < self._frac_part).float()

        indices = []
        for dataset_index, rep_factor in enumerate(rep_factors):
            indices.extend([dataset_index] * int(rep_factor.item()))
        return torch.tensor(indices, dtype=torch.int64)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self._seed + self._epoch)

        indices = self._get_epoch_indices(g)
        if self._shuffle:
            randperm = torch.randperm(len(indices), generator=g)
            indices = indices[randperm].tolist()
        else:
            indices = indices.tolist()

        # Pad so that indices can be evenly split across distributed ranks.
        if len(indices) % self._world_size:
            padding_size = self._world_size - len(indices) % self._world_size
            indices += indices[:padding_size]

        assert len(indices) % self._world_size == 0
        self.indices = indices[self._rank::self._world_size]

        return iter(self.indices)

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            indices = self._get_epoch_indices(g)
            if self._shuffle:
                randperm = torch.randperm(len(indices), generator=g)
                yield from indices[randperm].tolist()
            else:
                yield from indices.tolist()
