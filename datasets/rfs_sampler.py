import itertools
import torch
import math
from torch.utils.data.sampler import Sampler
from util.misc import all_gather, shared_random_seed
from util import misc
from collections import defaultdict


class RepeatFactorTrainingSampler(Sampler):
    """
    Similar to TrainingSampler, but a sample may appear more times than others based
    on its "repeat factor". This is suitable for training on class imbalanced datasets like LVIS.
    """

    def __init__(self, repeat_factors, *, shuffle=True, seed=None):
        """
        Args:
            repeat_factors (Tensor): a float vector, the repeat factor for each indice. When it's
                full of ones, it is equivalent to ``TrainingSampler(len(repeat_factors), ...)``.
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        self._shuffle = shuffle
        self._epoch = 0
        if seed is None:
            seed = shared_random_seed()
        self._seed = int(seed)

        self._rank = misc.get_rank()
        self._world_size = misc.get_world_size()

        repeat_factors = torch.as_tensor(repeat_factors, dtype=torch.float32)
        # Split into whole number (_int_part) and fractional (_frac_part) parts.
        self._int_part = torch.trunc(repeat_factors)
        self._frac_part = repeat_factors - self._int_part
        self.num_samples = int(math.ceil(float(repeat_factors.sum().item()) / self._world_size))
        self.total_size = self.num_samples * self._world_size

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch):
        self._epoch = epoch

    @staticmethod
    def repeat_factors_from_category_frequency(dataset_dicts, repeat_thresh):
        """
        Compute (fractional) per-image repeat factors based on category frequency.
        The repeat factor for an image is a function of the frequency of the rarest
        category labeled in that image. The "frequency of category c" in [0, 1] is defined
        as the fraction of images in the training set (without repeats) in which category c
        appears.
        See :paper:`lvis` (>= v2) Appendix B.2.

        Args:
            dataset_dicts (list[dict]): annotations in Detectron2 dataset format.
            repeat_thresh (float): frequency threshold below which data is repeated.
                If the frequency is half of `repeat_thresh`, the image will be
                repeated twice.

        Returns:
            torch.Tensor:
                the i-th element is the repeat factor for the dataset image at index i.
        """
        # 1. For each category c, compute the fraction of images that contain it: f(c)
        category_freq = defaultdict(int)
        for dataset_dict in dataset_dicts:  # For each image (without repeats)
            cat_ids = {ann["category_id"] for ann in dataset_dict["annotations"]}
            for cat_id in cat_ids:
                category_freq[cat_id] += 1
        num_images = len(dataset_dicts)
        for k, v in category_freq.items():
            category_freq[k] = v / num_images

        # 2. For each category c, compute the category-level repeat factor:
        #    r(c) = max(1, sqrt(t / f(c)))
        category_rep = {
            cat_id: max(1.0, math.sqrt(repeat_thresh / cat_freq))
            for cat_id, cat_freq in category_freq.items()
        }

        # 3. For each image I, compute the image-level repeat factor:
        #    r(I) = max_{c in I} r(c)
        rep_factors = []
        for dataset_dict in dataset_dicts:
            cat_ids = {ann["category_id"] for ann in dataset_dict["annotations"]}
            rep_factor = max({category_rep[cat_id] for cat_id in cat_ids}, default=1.0)
            rep_factors.append(rep_factor)

        return torch.tensor(rep_factors, dtype=torch.float32)

    def _get_epoch_indices(self, generator):
        """
        Create a list of dataset indices (with repeats) to use for one epoch.

        Args:
            generator (torch.Generator): pseudo random number generator used for
                stochastic rounding.

        Returns:
            torch.Tensor: list of dataset indices to use in one epoch. Each index
                is repeated based on its calculated repeat factor.
        """
        # Since repeat factors are fractional, we use stochastic rounding so
        # that the target repeat factor is achieved in expectation over the
        # course of training
        rands = torch.rand(len(self._frac_part), generator=generator)
        rep_factors = self._int_part + (rands < self._frac_part).float()
        # Construct a list of indices in which we repeat images as specified
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

        print(len(indices), self.total_size, self._rank, self._world_size, self.num_samples)
        if len(indices) % self._world_size:
            indices += indices[:(self._world_size - len(indices) % self._world_size)]

        assert len(indices) % self._world_size == 0
        self.indices = indices[self._rank::self._world_size]

        return iter(self.indices)

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            # Sample indices with repeats determined by stochastic rounding; each
            # "epoch" may have a slightly different size due to the rounding.
            indices = self._get_epoch_indices(g)
            if self._shuffle:
                randperm = torch.randperm(len(indices), generator=g)
                yield from indices[randperm].tolist()
            else:
                yield from indices.tolist()


class FixedLengthRepeatFactorTrainingSampler(Sampler):
    """
    Sample from repeat factors with a fixed epoch length.

    RepeatFactorTrainingSampler materializes roughly sum(repeat_factors) images
    per epoch. This sampler instead uses the repeat factors as sampling weights
    and draws a fixed number of images, defaulting to the original dataset size.
    It is useful when comparing RFS/IRFS against a no-sampling baseline with the
    same number of optimizer steps.
    """

    def __init__(
            self,
            repeat_factors,
            *,
            sample_size=None,
            sampling_power=1.0,
            replacement=True,
            seed=None,
    ):
        repeat_factors = torch.as_tensor(repeat_factors, dtype=torch.float32)
        if repeat_factors.dim() != 1 or repeat_factors.numel() == 0:
            raise ValueError("repeat_factors must be a non-empty 1-D tensor.")
        if sampling_power < 0:
            raise ValueError("sampling_power must be non-negative.")

        weights = repeat_factors.pow(float(sampling_power))
        if not torch.isfinite(weights).all() or weights.sum() <= 0:
            raise ValueError("repeat-factor weights must be finite and have positive sum.")
        self.weights = weights / weights.sum()

        self._epoch = 0
        if seed is None:
            seed = shared_random_seed()
        self._seed = int(seed)

        self._rank = misc.get_rank()
        self._world_size = misc.get_world_size()
        self._replacement = replacement

        dataset_size = int(repeat_factors.numel())
        if sample_size is None:
            sample_size = dataset_size
        self._global_size = int(sample_size)
        if self._global_size <= 0:
            raise ValueError("sample_size must be positive.")

        self.num_samples = int(math.ceil(self._global_size * 1.0 / self._world_size))
        self.total_size = self.num_samples * self._world_size

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self._epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self._seed + self._epoch)

        if self._replacement:
            indices = torch.multinomial(
                self.weights,
                self.total_size,
                generator=g,
                replacement=True,
            )
        else:
            if self.total_size > self.weights.numel():
                raise ValueError("replacement=False requires total_size <= dataset size.")
            indices = torch.multinomial(
                self.weights,
                self.total_size,
                generator=g,
                replacement=False,
            )

        indices = indices.tolist()
        indices = indices[:self.total_size]
        assert len(indices) == self.total_size
        indices = indices[self._rank:self.total_size:self._world_size]
        assert len(indices) == self.num_samples
        return iter(indices)


class IRFSTrainingSampler(RepeatFactorTrainingSampler):
    """
    Instance-Aware Repeat Factor Sampling.

    IRFS keeps the same image-level repeat-and-round procedure as RFS, but
    computes the category repeat factor from both image frequency and instance
    frequency:

        f_irfs(c) = sqrt(f_image(c) * f_instance(c))
        r(c) = max(1, sqrt(t / f_irfs(c)))
    """

    @staticmethod
    def repeat_factors_from_image_and_instance_frequency(dataset_dicts, repeat_thresh):
        image_freq = defaultdict(int)
        instance_freq = defaultdict(int)
        total_instances = 0

        for dataset_dict in dataset_dicts:
            annotations = dataset_dict.get("annotations", [])
            cat_ids = {ann["category_id"] for ann in annotations}
            for cat_id in cat_ids:
                image_freq[cat_id] += 1
            for ann in annotations:
                instance_freq[ann["category_id"]] += 1
                total_instances += 1

        num_images = max(len(dataset_dicts), 1)
        total_instances = max(total_instances, 1)
        category_rep = {}
        for cat_id, image_count in image_freq.items():
            box_count = instance_freq.get(cat_id, 0)
            if image_count <= 0 or box_count <= 0:
                continue
            freq_image = image_count / num_images
            freq_instance = box_count / total_instances
            freq_irfs = math.sqrt(freq_image * freq_instance)
            category_rep[cat_id] = max(1.0, math.sqrt(float(repeat_thresh) / freq_irfs))

        rep_factors = []
        for dataset_dict in dataset_dicts:
            cat_ids = {ann["category_id"] for ann in dataset_dict.get("annotations", [])}
            rep_factor = max((category_rep.get(cat_id, 1.0) for cat_id in cat_ids), default=1.0)
            rep_factors.append(rep_factor)

        return torch.tensor(rep_factors, dtype=torch.float32)


from typing import Optional


class InstanceAwareTrainingSampler(Sampler):
    """
    Sample images with replacement using weights accumulated from object instances.

    This is an instance-aware image sampler: the dataloader still returns full
    images, but an image with more tail instances receives more sampling mass.
    Unlike RepeatFactorTrainingSampler, the default epoch length is fixed to the
    original dataset length so training budgets are easier to compare.
    """

    def __init__(
            self,
            dataset_dicts,
            *,
            repeat_thresh=0.001,
            sample_size=None,
            object_penalty_power=0.5,
            replacement=True,
            seed: Optional[int] = None,
    ):
        self._epoch = 0
        if seed is None:
            seed = shared_random_seed()
        self._seed = int(seed)

        self._rank = misc.get_rank()
        self._world_size = misc.get_world_size()
        self._replacement = replacement

        dataset_size = len(dataset_dicts)
        if sample_size is None:
            sample_size = dataset_size
        self._global_size = int(sample_size)
        assert self._global_size > 0
        self.num_samples = int(math.ceil(self._global_size * 1.0 / self._world_size))
        self.total_size = self.num_samples * self._world_size

        self.weights = self.compute_image_weights(
            dataset_dicts,
            repeat_thresh=repeat_thresh,
            object_penalty_power=object_penalty_power,
        )
        if self.weights.numel() != dataset_size:
            raise ValueError("Instance-aware weights must match dataset length.")
        if not torch.isfinite(self.weights).all() or self.weights.sum() <= 0:
            raise ValueError("Instance-aware weights must be finite and have positive sum.")

    def set_epoch(self, epoch):
        self._epoch = epoch

    def __len__(self):
        return self.num_samples

    @staticmethod
    def compute_image_weights(dataset_dicts, repeat_thresh=0.001, object_penalty_power=0.5):
        """
        Build per-image sampling weights from all object instances in an image.

        Category weights follow the LVIS RFS category repeat rule using image
        frequency: w(c) = max(1, sqrt(t / f(c))). Image weights sum the weights
        of all instances and divide by num_objects ** object_penalty_power to
        reduce common/frequent co-occurrence amplification.
        """
        category_freq = defaultdict(int)
        for dataset_dict in dataset_dicts:
            cat_ids = {ann["category_id"] for ann in dataset_dict.get("annotations", [])}
            for cat_id in cat_ids:
                category_freq[cat_id] += 1

        num_images = max(len(dataset_dicts), 1)
        category_weight = {
            cat_id: max(1.0, math.sqrt(float(repeat_thresh) / (count / num_images)))
            for cat_id, count in category_freq.items()
            if count > 0
        }

        image_weights = []
        for dataset_dict in dataset_dicts:
            annotations = dataset_dict.get("annotations", [])
            if not annotations:
                image_weights.append(1.0)
                continue

            score = sum(category_weight.get(ann["category_id"], 1.0) for ann in annotations)
            penalty = max(1.0, float(len(annotations)) ** float(object_penalty_power))
            image_weights.append(score / penalty)

        return torch.tensor(image_weights, dtype=torch.float32)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self._seed + self._epoch)

        if self._replacement:
            indices = torch.multinomial(
                self.weights,
                self.total_size,
                generator=g,
                replacement=True,
            )
        else:
            if self.total_size > len(self.weights):
                raise ValueError("replacement=False requires total_size <= dataset size.")
            indices = torch.multinomial(
                self.weights,
                self.total_size,
                generator=g,
                replacement=False,
            )

        indices = indices.tolist()
        indices = indices[:self.total_size]
        assert len(indices) == self.total_size
        indices = indices[self._rank:self.total_size:self._world_size]
        assert len(indices) == self.num_samples
        return iter(indices)
