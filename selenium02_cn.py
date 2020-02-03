"""
GOOGLE IMAGE CRAWLER WITH SELENIUM
2020 01 30 JJH
MODIFIED FROM https://j-remind.tistory.com/61
The code somehow works well on Pycharm environment.
"""

from selenium import webdriver
import os
import urllib.request

# 찾고자 하는 검색어를 url로 만들어 준다.
searchterm = '恙虫病焦痂'
url = "https://image.so.com/i?q=" + searchterm + "&src=tab_www"

# chrom webdriver 사용하여 브라우저를 가져온다.
browser = webdriver.Chrome('C:/dev/final/image/chromedriver.exe')  # 각자 경로에 따라 변경 필
browser.get(url)

# User-Agent를 통해 봇이 아닌 유저정보라는 것을 위해 사용
# Chrome 주소창에 chrome://version 접속 후 사용자 에이전트 값을 찾아서 변경
header = {'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36"}

# 이미지 카운트 (이미지 저장할 때 사용하기 위해서)
counter = 0
succounter = 0

# 소스코드가 있는 경로에 '검색어' 폴더가 없으면 만들어준다.(이미지 저장 폴더를 위해서)
if not os.path.exists(searchterm):
    os.mkdir(searchterm)

for _ in range(1000):
    # 가로 = 0, 세로 = 10000 픽셀 스크롤한다.
    browser.execute_script("window.scrollBy(0,10000)")

# a 태그에서 class name이 entity인 것을 찾아온다
for x in browser.find_elements_by_xpath("//a[@class='entity']"):
    counter = counter + 1
    print("Total Count:", counter)
    print("Succsessful Count:", succounter)
    print("URL:", x.find_element_by_tag_name('img').get_attribute('src'))

    # 이미지 url
    img = x.find_element_by_tag_name('img').get_attribute('src')
    # 이미지 확장자
    imgtype = img[img.rfind(".")+1:]

    # 구글 이미지를 읽고 저장한다.
    try:
        # urllib.request가 python3부터는 module이 되었기때문에, 이 기능을 수행하는 class Request를 호출합니다.
        req = urllib.request.Request(img, headers=header)
        # 그리고 headers parameter는 I AM NOT ROBOT임을 인증하는 파트로, 이를 dict type인 header 변수를 사용하도록 고쳤습니다.
        raw_img = urllib.request.urlopen(req).read()
        File = open(os.path.join(searchterm, searchterm + "_" + str(counter) + "." + imgtype), "wb")
        File.write(raw_img)
        File.close()
        succounter = succounter + 1
    except:
        print("can't get img")

print(succounter, "succesfully downloaded")
browser.close()