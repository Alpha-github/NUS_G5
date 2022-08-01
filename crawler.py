# Using Bing Image Search API to find image dataset

import requests
import matplotlib.pyplot as plt


subscription_key = "" # API KEY
search_url = "https://api.bing.microsoft.com/v7.0/images/search"
search_term = ["tomato","potato","banana","onion","apple"]

headers = {"Ocp-Apim-Subscription-Key" : subscription_key}

for vegie in search_term:
    search = "rotten "+vegie
    params  = {"q": search, "license": "public", "imageType": "photo"}
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()
    thumbnail_urls = [img["thumbnailUrl"] for img in search_results["value"][:500]]

    for url in range(len(thumbnail_urls)):
        img_data = requests.get(thumbnail_urls[url]).content
        with open('.\\vegies\\rotten\\{veg}\\{i}.jpg'.format(i=url,veg=vegie), 'wb') as handler:
            handler.write(img_data)