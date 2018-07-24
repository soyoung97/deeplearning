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
fw = h5py.File('/home/test/deeplearning/GrassFile.h5','w')
input_binary = np.zeros([5100, 112, 112, 3])
input_color = np.zeros([5100, 112, 112, 3])
next_offset = 0
idx = 0
#picnic = 573
#view = 489
#waterfall = 445
#desert = 400
#national_park = 374
#island = 350
#beach = 336
#city = 294
#sand = 289
##ocean = 254
#farm = 246
#landscape = 242
#la = 239
#building = 222
#forest = 222
#historic sites = 215
#scenery = 205
#jungle = 154
#nature = 150
#grassland = 122


search_term = ["picnic","picnic","picnic","picnic","view","view","view","waterfall","waterfall","waterfall","desert","desert","national park","national park","island","island","beach","beach","city","city","sand","sand","farm","farm","landspace","landscape","building","forest","historic sites", "scenery", "jungle", "nature","grassland","la"]
for iterations in range(len(search_term)):
    if (iterations != 0) and (search_term[iterations] == search_term[iterations -1]):
        off = next_offset
    else:
        off = 0
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
        image = image.resize((112,112))
        #settings for saving color image
        numpy_img = np.array(image)
        input_color[idx] = numpy_img
        
        pix = image.load()
       # string = "/home/test/madcamp/portrait/image"+str(i)+"_c.jpeg"
       # image.save(string, "jpeg")
        binary_arr = np.zeros([112, 112])
        #settings for saving binary image
        for x in range(112):
            for y in range(112):
                (a,b,c) = pix[x,y]
                r = (a+ b + c) // 3
                #binary_arr[x,y] = r
                pix[x,y] = (r,r,r)

        tf_numpy_img = np.array(image)
        input_binary[idx] = tf_numpy_img
        #string_b = "/home/test/madcamp/portrait/image"+str(i)+"_b.jpeg"
       # binary_arr.save(string_b,"jpeg")
        #image.save(string_b, "jpeg")
        idx += 1

fw.create_dataset('binary_img', data=input_binary)
fw.create_dataset('color_img', data=input_color)
fw.close()

#bi = fr.get('binary_img')
#ci = fr.get('color_img')
#bi[0:100]
#ci[0:100]
