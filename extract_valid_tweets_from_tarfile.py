import tarfile
import bz2, json, io
import os.path
interest_loc = ['Boston, MA', 'New York, NY', 'Atlanta, GA','Orlando, FL','Seattle, WA','Portland, OR','San Diego, CA','Phoenix, AZ','St. Louis, MO','Kansas City, KS']

def loc_from_content(content, file_name):
    mode = "w"
    if os.path.isfile(file_name):
        mode = "a"
    with open(file_name, mode) as text_file:
        for line in (content):
            if 'location' not in line:
                continue
            d = json.loads(line)
            for i_loc in interest_loc:
                if(d['user']['location'] == i_loc):
                    text_file.write(line+'\n')
                    break
    return

root = '2016-03'
tar = tarfile.open("archiveteam-twitter-stream-"+root+".tar",'r')
file = 0
file_number = 0
for member in tar.getmembers():
    if(member.name[-3:] != "bz2"):
        continue
    print(member.name)
    f = tar.extractfile(member)
    content = f.read()
    content = bz2.decompress(content)
    content = content.splitlines()

    file_number += 1
    if(file_number % 100 == 0):
        file_number = 0
        file += 1

    loc_from_content(content, file_name = root+'/'+str(file)+".json")
    #break
tar.close()