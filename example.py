import ProbIR as PIR
from tqdm import tqdm
import csv
import pickle

data = []
with open("./moviesummaries/wiki_movie_plots_deduped_with_summaries.csv", 'r',  encoding="utf8",) as f:
    reader = csv.reader(f, dialect='excel-tab', delimiter=',' )
    for row in tqdm(reader):
        data.append(row)

movies = []
for elem in data:
    tmp = PIR.Document(elem[1], elem[7])
    movies.append(tmp)

with open('moviesummaries/idx.pkl', 'rb') as f:
    idx = pickle.load(f)
with open('moviesummaries/tf.pkl', 'rb') as f:
    tf = pickle.load(f)
with open('moviesummaries/idf.pkl', 'rb') as f:
    idf = pickle.load(f)

IR = PIR.ProbIR(movies,idx,tf,idf)

print("Insert query to submit: ")
qry = input()
ans = IR.query(qry, results=10, pseudorel=20)
print("Final retrived documents:")
PIR.ordered_print(ans)