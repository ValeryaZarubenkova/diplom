import csv
import re
import pandas as pd
import nltk
import emoji
from nltk.tokenize import word_tokenize 
from csv import writer
from pymystem3 import Mystem
from nltk.corpus import stopwords

def f_resub(file_first, file_second):
    j = 0
    w = 0
    alltexts = ''
    dobmas = [None] * 10000
    lemmatizer = Mystem()
    stopwordss = stopwords.words("russian")
    stopwordss.remove('хорошо')
    stopwordss.remove('не')
    stopwordss.remove('ни')
    def largest_divisor(n):
        divisor = n / 2
        while n % divisor != 0:
            divisor -= 1
        return divisor
    with open(file_first,'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        num_lines = len(list(reader))
        print(num_lines)
        divisor = int(largest_divisor(num_lines))
        print(divisor)
        with open(file_first,'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            with open(file_second,'w', encoding='utf-8') as sec_file: 
                    sec_file.truncate(0)
                    file_writer = csv.writer(sec_file, delimiter = ",", lineterminator="\r")
                    file_writer.writerow(['ozenka','comments']) 
                    for row in reader:
                        listik = []
                        dob1_0 = row[1]
                        dob2 = re.sub(r'[a-zA-Z]', '', dob1_0)
                        dob2 = re.sub(u'[Ёё]', u'е', dob2) 
                        dob2 = re.sub(r'[^\w\s\U0001F000-\U0001F9FF]', ' ', dob2)
                        dob2 = re.sub(r'\s+', ' ', dob2)
                        dob2 = re.sub(r'\d+', '', dob2)
                        dob2 = re.sub(r'u[d200]+', '', dob2)
                        dob2 = dob2.replace('🏻', '').replace('🏼', '').replace('🏽', '').replace('_', '') 
                        dob2 = dob2.lower()
                        dob = int(row[0]) 
                        if dob >= 4:
                            dobmas[j] = 1
                        else:
                            dobmas[j] = 0
                        
                        j = j+1 #счетчик для отмерки корличества объединяемых отзывов
                        alltexts = alltexts + dob2 + ' br '   #записываю в одну строку все отзывы с разделителем  ' br ' между ними  
                        if j == divisor:
                            res = []
                            words = lemmatizer.lemmatize(alltexts)
                            doc = []
                            for txt in words:  #прохожусь по словам в полученном после лемматизации списке
                                if txt != '\n' and txt.strip() != '' and txt not in stopwordss and ((len(txt) > 1 and txt.isalpha()) or not txt.isalpha()):
                                    if txt == 'br':
                                        res.append(doc)
                                        file_writer.writerow([dobmas[w],res[0]]) 
                                        w = w+1
                                        print(w)
                                        doc = []
                                    else:
                                        doc.append(txt)
                                        res = []
                            alltexts = ''
                            w = 0
                            j = 0
