from typing import List, Tuple, Optional, Union
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from T2IBenchmark.feature_extractors import BaseFeatureExtractor, InceptionV3FE
from T2IBenchmark.loaders import BaseImageLoader, ImageDataset, get_images_from_folder, validate_image_paths
from T2IBenchmark.metrics import FIDStats, frechet_distance
from T2IBenchmark.utils import dprint, set_all_seeds


def create_dataset_from_input(obj: Union[str, List[str], BaseImageLoader]) -> Union[BaseImageLoader, FIDStats]:
    if isinstance(obj, str):
        if obj.endswith('.npz'):
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
    verbose: bool = True
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
    device: torch.device = 'cuda',
    seed: Optional[int] = 42,
    batch_size: int = 128,
    dataloader_workers: int = 16,
    verbose: bool = True
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
                dataset, batch_size=batch_size, shuffle=False, 
                drop_last=False, num_workers=dataloader_workers
            )
            features = get_features_for_dataset(dataloader, inception_fe, verbose=verbose)
            all_features.append(features)
            stats.append(FIDStats.from_features(features))
        else:
            raise NotImplementedError()
            
    fid = frechet_distance(stats[0], stats[1])
    dprint(verbose, f"FID is {fid}")
    return fid, (
        {'features': all_features[0], 'stats': stats[0]},
        {'features': all_features[1], 'stats': stats[1]}
    )
