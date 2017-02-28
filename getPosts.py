from urllib import urlopen
from bs4 import BeautifulSoup
import subprocess
import glob
import os
import sys
import re


def getEnglish(fname, url, mdir="data"):
    try:
        html = urlopen(url).read()
    except:
        "Failed to open: %s, %s" %(fname, url)
        return 
    soup = BeautifulSoup(html, "html5lib")
    txt = soup.findAll("div", {"class": "wiki-content"})

    mTxt = ''
    if len(txt) > 0:
        mTxt = txt[0]
    else:
        return
    f = open(os.path.join(mdir, fname), "w")
    f.close()

    #mSoup = BeautifulSoup(mTxt.get_text(), 'html5lib')
    paragraphs = [x for x in mTxt.findAll('p')]
    #print len(paragraphs)
    #sys.exit(1)
    for mTxt in paragraphs:
        with open("blah1.txt", "w") as fp:
            lTxt = mTxt.get_text().encode('utf-8')
            lTxt = re.sub(r"http\S+", "", lTxt)
            fp.write(lTxt)

        try:
            #transytext = subprocess.check_output("trans -e bing file://blah1.txt", shell=True)
            #transytext = subprocess.check_output("trans -e yandex file://blah1.txt", shell=True)
            p = subprocess.Popen("trans -e bing file://blah1.txt", shell=True, stdout=subprocess.PIPE)
            p.wait()
            transytext = p.communicate()[0]

            with open(os.path.join(mdir, fname), "a") as fp:
                fp.write(transytext)
        except:
            print "Failed %s %s" % (fname, url)



subDir = sys.argv[1]
users = glob.glob('urls/' + subDir + '/*.out')

for userFile in users:
    user = os.path.basename(userFile.split('_')[0])
    with open(userFile) as fp:
        for i, line in enumerate(fp):
            url = line.strip()
            fname = "%s_%d.txt" % (user, i)
            print "%s: %s" % (fname, url)
            getEnglish(fname, url, mdir='data/' + subDir)
