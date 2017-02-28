import re
import sys
blackList = {}
#/blog
#/handbook/Jobs
#/handbook/Help
#/handbook/Service-status
#/privacy
#/terms
#/privacy/
#/privacy/
user = 'dummy'
haveUser = False
for line in sys.stdin:
    m = re.search('href=\"(.*?)\"', line)
    if m and 'users' in m.group(1) and 'notebook/' in m.group(1):
        #print m.group(1)
        mm = re.search('users\/.*?\/', m.group(1))
        if mm and not haveUser:
            user = mm.group(0).split('/')[1]
            print user
            haveUser = True
        if haveUser:
            with open("urls/%s_urls.out" % user, 'a') as fp:
                fp.write(m.group(1) + '\n')
