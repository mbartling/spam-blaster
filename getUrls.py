from urllib import urlopen
from bs4 import BeautifulSoup
import sys
import os
import re

mFile  = sys.argv[1]  # file containing notebook urls
subDir = sys.argv[2]  # spam/not-spam

with open(mFile) as fp:
    for mUrl in fp:
        if not mUrl:
            continue
        mUrl = mUrl.strip()
        m = re.search("users\/(.*?)\/", mUrl)
        user = 'dummy'
        if m:
            user = m.groups(1)
        html = urlopen(mUrl).read()
        soup = BeautifulSoup(html, "html5lib")
        
        with open(os.path.join("urls", subDir, "%s_urls.out" % user), "w") as op:
            for a in soup.find_all('a', href=True):
                if 'notebook/' in a['href'] and 'login' not in a['href']:
                    op.write("https://developer.mbed.org" + a['href'] + "\n")

