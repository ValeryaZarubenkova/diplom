import requests
import json
import pandas as pd
from fake_useragent import UserAgent 
import csv

def f_parsing(artickle, file_first):
    with_artickle = 'https://card.wb.ru/cards/detail?nm='+str(artickle)
    response = requests.get(with_artickle, headers={'User-Agent': UserAgent().chrome})
    reply = response.json()["data"]["products"][0]["root"]

    url = 'https://feedbacks1.wb.ru/feedbacks/v1/'+ str(reply)
    response = requests.get(url, headers={'User-Agent': UserAgent().chrome}) 

    if response.json()['feedbackCount']!=0:
        with open(file_first, 'a', encoding="utf-8", newline='') as f:
            f.truncate(0)   #очищение файла
            for item in response.json()['feedbacks']:
                print(item['text'], item['productValuation'])              
                writer = csv.writer(f)
                ozenka = repr(item['productValuation']) 
                comments = repr(item['text'].replace("\n", " ")) 
                writer.writerow([ozenka,comments])