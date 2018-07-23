import pdb
import matplotlib.pyplot as plt
import numpy as np
import h5py
import requests
import os
import shutil
import urllib
from io import BytesIO
from PIL import Image
import numpy as np

subscription_key2 = '5a76a2c660854e9f96726741d58ccfe4'
subscription_key1 = '10ca1116daba486e9e6ba229a1a9f9e4'

search_url = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"

headers = {"Ocp-Apim-Subscription-Key" : subscription_key1}

#settings for h5py
fw = h5py.File('/home/test/madcamp/portrait/myfile.h5','w')
input_binary = np.zeros([3000, 224, 224, 3])
input_color = np.zeros([3000, 224, 224, 3])
next_offset = 0
idx = 0
for iterations in range(20):
    search_term = ["selfie", "selfie", "selfie", "selfie", "portrait", "portrait", "face", "face", "people", "people", "person", "person","man", "man", "women", "women", "boy", "boy", "girl", "girl"]
    if iterations <= 3:
        off = next_offset
    else:
        off = (iterations % 2)*150
    params  = {"q": search_term[iterations], "license": "public", "imageType": "photo","count":"150", "offset": off, "color":"ColorOnly"}
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()
    next_offset = search_results["nextOffset"]
    est_mat = search_results["totalEstimatedMatches"]
    print("iterations : "+str(iterations)+". search_term = "+search_term[iterations]+" est-mat : "+str(est_mat))
    thumbnail_urls = [img["thumbnailUrl"] for img in search_results["value"]]
    for i in range(len(thumbnail_urls)):
        image_data = requests.get(thumbnail_urls[i])
        image_data.raise_for_status()
        image = Image.open(BytesIO(image_data.content))
        image = image.resize((224,224))
        #settings for saving color image
        numpy_img = np.array(image)
        input_color[idx] = numpy_img
        
        pix = image.load()
        string = "/home/test/madcamp/portrait/image"+str(i)+"_c.jpeg"
        image.save(string, "jpeg")
        binary_arr = np.zeros([224, 224])
        #settings for saving binary image
        for x in range(224):
            for y in range(224):
                (a,b,c) = pix[x,y]
                r = (a+ b + c) // 3
                #binary_arr[x,y] = r
                pix[x,y] = (r,r,r)

        tf_numpy_img = np.array(image)
        input_binary[idx] = tf_numpy_img
        string_b = "/home/test/madcamp/portrait/image"+str(i)+"_b.jpeg"
       # binary_arr.save(string_b,"jpeg")
        image.save(string_b, "jpeg")
        idx += 1

fw.create_dataset('binary_img', data=input_binary)
fw.create_dataset('color_img', data=input_color)
fw.close()

#bi = fr.get('binary_img')
#ci = fr.get('color_img')
#bi[0:100]
#ci[0:100]
