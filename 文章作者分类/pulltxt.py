# encoding: utf-8
"""
@author: 徐瑞
@time: 2020/9/21 23:58
@file: classification.py
@desc: 
"""
import requests
from bs4 import BeautifulSoup

# 采集网页中的文章信息
for i in range(1, 3):
    string_text = ''
    if i < 10:
        url = 'https://avalon.law.yale.edu/subject_menus/fed0' + str(i) + '.asp'
        re = requests.get(url)
        soup = BeautifulSoup(re.text, "html5lib")
        content = soup.find_all(name='p')
        for t in content:
            string_text += t.text
        # string_text = ''.join(string_text)
        fh = open('文章.txt', 'w', encoding='utf-8')
        fh.write(string_text)
        fh.close()
        print(soup)
    else:
        url = 'https://avalon.law.yale.edu/subject_menus/fed' + str(i) + '.asp'
        re = requests.get(url)
        soup = BeautifulSoup(re, 'html5lib')
        content = soup.find_all('p')
        string_text = [t.text for t in content]
        string_text = ''.join(string_text)
        fh = open('文章.txt', 'w', encoding='utf-8')
        fh.write(string_text)
        fh.close()
        # print(string_text)
