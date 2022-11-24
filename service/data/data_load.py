from pathlib import Path
import requests
import click
import zipfile as zf
from tqdm.auto import tqdm


@click.command()
@click.argument("output_path", type=click.Path())
def download_kion_dataset(
    output_path: str
):
    """
    Save kion data yandexcloud urls
    :param output_path: path to save data
    """
    url = "https://storage.yandexcloud.net/itmo-recsys-public-data/kion_train.zip"

    req = requests.get(url, stream=True)
    
    output_path = Path(output_path).resolve()
    save_zip_path = output_path / "kion_train.zip"

    with open(save_zip_path, "wb") as fd:
        total_size_in_bytes = int(req.headers.get('Content-Length', 0))
        progress_bar = tqdm(desc='kion dataset download', total=total_size_in_bytes, unit='iB', unit_scale=True)
        for chunk in req.iter_content(chunk_size=2 ** 20):
            progress_bar.update(len(chunk))
            fd.write(chunk)
    
    files = zf.ZipFile(save_zip_path,'r')
    files.extractall(path=output_path)
    files.close()
    
    
if __name__ == "__main__":
    download_kion_dataset()