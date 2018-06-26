#!/usr/bin/env python
#-*- coding:utf-8 -*-

import urllib.request
import sys
import json
import os
import yaml

f = open("config.yaml", "r+")
cfg = yaml.load(f)
GOOGLE_CLOUD_STORAGE_DIR = cfg['GOOGLE_CLOUD_STORAGE_DIR']
checkpoints = cfg['checkpoints']
chk = cfg['chk']

def download(chkpoint,filename):

    url = os.path.join(GOOGLE_CLOUD_STORAGE_DIR , chkpoint , filename)

    urllib.request.urlretrieve(url, os.path.join('./waits/' , chkpoint , filename))

if __name__ == "__main__":

    chkpoint = checkpoints[chk]
    print(chkpoint)
    save_dir = './waits/'+ chkpoint
    print(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    download(chkpoint, 'manifest.json')

    f = open(os.path.join(save_dir,'manifest.json'), 'r')
    json_dict = json.load(f)

    for x in json_dict:
        filename = json_dict[x]['filename']
        print(filename)
        download(chkpoint,filename)
