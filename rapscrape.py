from urlparse import urljoin
from BeautifulSoup import BeautifulSoup
import requests


BASE_URL = "http://genius.com"
artist_url = "http://genius.com/artists/Andre-3000/"

#response = requests.get(artist_url, headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.153 Safari/537.36'})

response = requests.get(artist_url)
#response.encoding = "utf-8"

text = response.text

soup = BeautifulSoup(text)
#print soup
print soup.select("song_title")

#for song_link in soup.select("ul.song_list > li > a"):
#    link = urljoin(BASE_URL, song_link['href'])
#    response = requests.get(link)
#    text = response.text
#    soup = BeautifulSoup(text)
#    lyrics = soup.find('div', class_='lyrics').text.strip()

    # tokenize `lyrics` with nltk