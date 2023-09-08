import os
from typing import Dict, List, Optional, Tuple, Union

import clip
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from T2IBenchmark.feature_extractors import BaseFeatureExtractor, InceptionV3FE
from T2IBenchmark.loaders import (
    BaseImageLoader,
    CaptionImageDataset,
    ImageDataset,
    get_images_from_folder,
    validate_image_paths,
)
from T2IBenchmark.metrics import FIDStats, frechet_distance
from T2IBenchmark.utils import dprint, set_all_seeds


def create_dataset_from_input(
    obj: Union[str, List[str], BaseImageLoader]
) -> Union[BaseImageLoader, FIDStats]:
    if isinstance(obj, str):
        if obj.endswith(".npz"):
            # fid statistics
            return FIDStats.from_npz(obj)
        else:
            # path to folder
            image_paths = get_images_from_folder(obj)
            dataset = ImageDataset(image_paths)
            return dataset
    elif isinstance(obj, list):
        # list of paths
        validate_image_paths(obj)
        dataset = ImageDataset(obj)
        return dataset
    elif isinstance(obj, BaseImageLoader):
        return obj
    else:
        raise ValueError(f"Input {obj} has unknown type. See the documentation")


def get_features_for_dataset(
    dataset: BaseImageLoader,
    feature_extractor: BaseFeatureExtractor,
    verbose: bool = True,
) -> np.ndarray:
    features = []
    for x in tqdm(dataset):
        feats = feature_extractor.forward(x).numpy()
        features.append(feats)

    res_feats = np.concatenate(features)
    return res_feats


def calculate_fid(
    input1: Union[str, List[str], BaseImageLoader],
    input2: Union[str, List[str], BaseImageLoader],
    device: torch.device = "cuda",
    seed: Optional[int] = 42,
    batch_size: int = 128,
    dataloader_workers: int = 16,
    verbose: bool = True,
) -> (int, Tuple[dict, dict]):
    if seed:
        set_all_seeds(seed)

    input1 = create_dataset_from_input(input1)
    input2 = create_dataset_from_input(input2)

    # create inception net
    inception_fe = InceptionV3FE(device)

    stats = []
    all_features = []
    # process inputs
    for input_data in [input1, input2]:
        dprint(verbose, f"Processing: {input_data}")
        if isinstance(input_data, FIDStats):
            all_features.append([])
            stats.append(input_data)
        elif isinstance(input_data, ImageDataset):
            # if a dataset-like
            dataset = input_data
            dataset.preprocess_fn = inception_fe.get_preprocess_fn()
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=dataloader_workers,
            )
            features = get_features_for_dataset(
                dataloader, inception_fe, verbose=verbose
            )
            all_features.append(features)
            stats.append(FIDStats.from_features(features))
        else:
            raise NotImplementedError()

    fid = frechet_distance(stats[0], stats[1])
    dprint(verbose, f"FID is {fid}")
    return fid, (
        {"features": all_features[0], "stats": stats[0]},
        {"features": all_features[1], "stats": stats[1]},
    )


def calculate_clip_score(
    image_paths: List[str],
    captions_mapping: Dict[str, str],
    device: torch.device = "cuda",
    seed: Optional[int] = 42,
    batch_size: int = 128,
    dataloader_workers: int = 16,
    verbose: bool = True,
):
    if seed:
        set_all_seeds(seed)

    model, preprocess = clip.load("ViT-B/32", device=device)
    dataset = CaptionImageDataset(
        images_paths=image_paths,
        captions=list(map(lambda x: captions_mapping[x], image_paths)),
        preprocess_fn=preprocess,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=dataloader_workers,
    )

    score_acc = 0.0
    num_samples = 0.0
    logit_scale = model.logit_scale.exp()

    for image, caption in tqdm(dataloader):
        image_embedding = model.encode_image(image.to(device))
        caption_embedding = model.encode_text(clip.tokenize(caption).to(device))

        image_features = image_embedding / image_embedding.norm(dim=1, keepdim=True).to(
            torch.float32
        )
        caption_features = caption_embedding / caption_embedding.norm(
            dim=1, keepdim=True
        ).to(torch.float32)

        score = logit_scale * (image_features * caption_features).sum()
        score_acc += score
        num_samples += image.shape[0]

    clip_score = score_acc / num_samples
    dprint(verbose, f"CLIP score is {clip_score}")

    return clip_score
