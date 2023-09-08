from typing import Dict
import pandas as pd
from T2IBenchmark.utils import download_and_cache_file
from T2IBenchmark.metrics import FIDStats

from .dataset import COCOImageDataset


COCO_CAPTIONS_URL = 'https://github.com/boomb0om/text2image-benchmark/releases/download/v0.0.1/MS-COCO_val2014_30k_captions.csv'

COCO_FID_STATS_URL = 'https://github.com/boomb0om/text2image-benchmark/releases/download/v0.0.1/MS-COCO_val2014_fid_stats.npz'


def get_coco_fid_stats() -> FIDStats:
    filepath = download_and_cache_file(COCO_FID_STATS_URL)
    return FIDStats.from_npz(filepath)


def get_coco_30k_captions() -> Dict[int, str]:
    filepath = download_and_cache_file(COCO_CAPTIONS_URL)
    df = pd.read_csv(filepath)
    return {i['image_id']: i['text'] for i in df.to_dict('records')}
