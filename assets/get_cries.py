#!/usr/bin/env python3

"""Script to download cries """
import json
import os
import requests
from tqdm import tqdm

asset_path = os.path.dirname(os.path.realpath(__file__))
output_asset_path = os.path.join(asset_path, "cries")
CRIES_URL = "https://pokemoncries.com/cries-old"

def get_file(fpath:str, pname:str):
    """ Download file and change name """
    data = requests.get(fpath)
    cry_path = os.path.join(output_asset_path, f"{pname}.mp3")
    with open(cry_path, 'wb') as f_name:
        f_name.write(data.content)


def download_all():
    """ Download all cries and change to names """
    print(f"Gotta cache 'em all!")

    pdict_path = os.path.join(asset_path, "names.json")
    with open(file=pdict_path, encoding='utf-8', mode='r') as f_name:
        pdict = json.load(f_name)

    # good gens only
    for idx in tqdm(range(0, 251)):
        p_url = os.path.join(CRIES_URL, f"{idx + 1}.mp3")
        pname = pdict[idx]
        get_file(p_url, pname)


if __name__ == "__main__":
    download_all()
