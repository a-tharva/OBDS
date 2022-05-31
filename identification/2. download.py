from pandas import read_csv
from urllib.request import urlretrieve
from os import listdir, mkdir, path as os_path
from tqdm import tqdm
from time import sleep
import requests

df = read_csv("birds.csv")

#Removing recordings that are not classified
df = df[df['en'] != 'Identity unknown'].copy()
df = df[df['en'] != 'Soundscape'].copy()
#Dropping files of species with less than 100 samples
counts = df['en'].value_counts()
chosen = counts.head(10).index
df = df[df["en"].isin(chosen)]
df.to_csv("to_download.csv", index=False)

print(df.shape)
print(len(df["en"].unique()), "different species")

audio_dir = 'audio_files/'
if not os_path.isdir(audio_dir):
    mkdir(audio_dir)

nots = []

for row in tqdm(df.iterrows(), total=df.shape[0]):
    url = row[1]["file"]
    f_name = str(row[1]["id"]) + row[1]["file-name"][-4:].lower()
    try:
        #urlretrieve('https:'+url, filename=audio_dir + f_name)
        filename = audio_dir + f_name
        r = requests.get(url, allow_redirects=True)
        open(filename, 'wb').write(r.content)
    except Exception as e:
        print("\nRetrying:", url)
        print(e) 
        sleep(60)
        try:
            urlretrieve('https:'+url, filename=audio_dir + f_name)
        except Exception as ee:
            print("Not downloaded|", f_name)
            print(ee)
            nots.append(row[1]["id"])
            pass

if len(nots) > 0:
    with open('not_downloaded.txt', 'w') as f:
        for item in nots:
            f.write(str(item) + '\n')
    print(str(nots))
else:
    print('All files were successfully downloaded!')