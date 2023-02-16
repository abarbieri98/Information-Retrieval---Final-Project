import ProbIR as PIR
import csv
import pickle

data = []
print("Loading data...")
with open("./moviesummaries/wiki_movie_plots_deduped_with_summaries.csv", 'r',  encoding="utf8",) as f:
    reader = csv.reader(f, dialect='excel-tab', delimiter=',' )
    for row in reader:
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
# IR = PIR.ProbIR.from_corpus() # nedded for custom corpora
print("Insert query to submit: ")
qry = input()
ans = IR.query(qry, results=10)
print("\nFinal retrived documents:")
PIR.ordered_print(ans)