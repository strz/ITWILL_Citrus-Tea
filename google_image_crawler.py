import urllib
from urllib.request import urlopen, Request   # URL 요청을 위한 클래스나 함수들이 정의
from urllib.parse import quote_plus  # URL의 구문을 분석하기 위한 함수들이 정의
from bs4 import BeautifulSoup

baseUrl = 'https://www.google.com.my/search?q='
plusUrl = input('검색어를 입력하세요:')

url = baseUrl + quote_plus(plusUrl) + '&tbm=isch'      # quote_plus: 웹에서 한글을 아스키 코드로 변환시켜줌.
# url = baseUrl + quote_plus(plusUrl) # quote_plus: 웹에서 한글을 아스키 코드로 변환시켜줌.
print(url)

req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})   # bot으로 오인하는 문제 해결하기 위해 헤더 넣어줌.(stackoverflow)
html = urlopen(req).read()
# text = response.decode('utf-8')
# print(text)

# html = urlopen(url).read()
soup = BeautifulSoup(html, 'html.parser')   # 분석해주는 것임.
img = soup.find_all(class_='rg_ic rg_i')

print(img)

# n = 1
# for i in img:
#     imgUrl = i['data-source']
#     with urlopen(imgUrl) as f:
#         with open(plusUrl + str(n) + 'jpg', 'wb') as h:
#             img = f.read()
#             h.write(img)
#     n += 1
#     print(imgUrl)
#
# print('다운로드 완료')
