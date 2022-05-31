import pandas as pd
import requests
from tqdm import tqdm

country = 'india'
response = requests.get(f'https://www.xeno-canto.org/api/2/recordings?query=cnt:{country}')
js = response.json()

df = pd.DataFrame()
for n_page in tqdm(range(1, js["numPages"]+1)):
    response = requests.get(f'https://www.xeno-canto.org/api/2/recordings?query=cnt:{country}&page={n_page}')
    page_json = response.json()
    ids, files, file_names, ens, lengths, gens = [], [], [], [], [], []
    for recording in page_json["recordings"]:
        ids.append(recording["id"])
        files.append(recording["file"])
        file_names.append(recording["file-name"])
        ens.append(recording["en"])
        lengths.append(recording["length"])
        gens.append(recording["gen"])


    df_ = pd.DataFrame.from_records({'id': ids, "file": files, "file-name": file_names, "en": ens, "gen": gens, "length": lengths})
    df = df.append(df_)
    
df.to_csv("birds.csv", index=False)