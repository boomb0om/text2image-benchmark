import os
import requests


def dprint(verbose: bool, *args):
    if verbose:
        print(*args)
        
        
def download_and_cache_file(url: str) -> str:
    home_dir = os.path.expanduser("~")
    cache_dir = os.path.join(home_dir, ".cache", 'T2IBenchmark')
    os.makedirs(cache_dir, exist_ok=True)

    filename = url.split("/")[-1]

    cache_path = os.path.join(cache_dir, filename)
    if os.path.exists(cache_path):
        return cache_path

    # Download the file from the URL
    response = requests.get(url)
    if response.status_code == 200:
        with open(cache_path, "wb") as file:
            file.write(response.content)
        return cache_path
    else:
        return None
