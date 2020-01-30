#-*- coding: utf-8 -*-

import os
from selenium import webdriver
import time
from urllib.request import urlretrieve


keyword = "scrub typhus"

# 웹 접속
print('접속 중')
driver = webdriver.Chrome('C:/chromedriver.exe')
driver.implicitly_wait(30)
url = 'https://search.naver.com/search.naver?where=image&sm=tab_jum&query={}'.format(keyword)
driver.get(url)

imgs = driver.find_elements_by_css_selector('img._image_source')
result = []
for img in imgs:
    if 'https' in img.get_attribute('src'):
        result.append(img.get_attribute('src'))
print(result)
#
# driver.close()
# print('수집 완료')
#
# # 폴더 생성
# # os.mkdir('./{}'.format(keyword))
#
# # 다운로드
# for index, link in enumerate(result):
#     start = link[0].rfind('.')
#     end = link[0].rfind('&')
#     print(result[0][start:end])