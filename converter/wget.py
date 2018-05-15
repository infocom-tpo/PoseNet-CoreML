#!/usr/bin/env python
#-*- coding:utf-8 -*-

import urllib.request
import sys
import json

def download(filename):

    url = "https://storage.googleapis.com/tfjs-models/weights/posenet/mobilenet_v1_101/" + filename

    urllib.request.urlretrieve(url,'./waits/'+filename)

if __name__ == "__main__":

    f = open('manifest.json', 'r')
    json_dict = json.load(f)

    for x in json_dict:
        filename = json_dict[x]['filename']
        print(filename)
        download(filename)
