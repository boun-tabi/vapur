from flask import Flask, render_template, request
import json
import operator
from collections import defaultdict, Counter

app = Flask(__name__)
with open("Query_Data/relations.json", "r") as f:
    relations = json.load(f)
with open("Query_Data/text_mapping.json", "r") as f:
    mappings = json.load(f)
with open("Query_Data/trigram_mappings.json", "r") as f:
    tri_mappings_l = json.load(f)

trigram_mappings = {}
for k, v in tri_mappings_l.items():
    trigram_mappings[k] = set(v)

del tri_mappings_l




@app.route('/')
def home():
    return render_template('search.html')


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/search/results', methods=['GET', 'POST'])
def search_request():
    search_term = request.form["input"]
    print(search_term)
    if search_term.lower().strip() in mappings:
        uid = mappings[search_term.lower().strip()][0]
        old = relations[uid]
        res = old.copy()
        res["Old Query"] = search_term
    else:
        new_query = get_similar(search_term.lower().strip())
        if new_query == "":
            uid = None
            res = {"Number of Mentions":0, "Old Query":search_term}
        else:
            uid = mappings[new_query.lower().strip()][0]
            old = relations[uid]
            res = old.copy()
            res["Old Query"] = search_term
            res["New Query"] = new_query

    return render_template('results.html', res=res)

def gjc(word, query):
    w3 = Counter([word[i:i+3] for i in range(len(word)-2)])
    q3 = Counter([query[i:i+3] for i in range(len(query)-2)])
    p = 0
    pd = 0
    for key in set(w3.keys()).union(q3.keys()):
        p += min(w3[key], q3[key])
        pd += max(w3[key], q3[key])
    
    if pd==0:
        return 0
    else:
        return p/pd

def get_similar(query):
    words = set()
    for g in [query[i:i+3] for i in range(len(query)-2)]:
        if g in trigram_mappings:
            words.update(trigram_mappings[g])

    if len(words)==0:
        return ""
    
                    
    best_score = 0
    best_val = ""
    for word in words:
        sc = gjc(word, query)
        print(word, sc)
        if sc>best_score:
            best_score = sc
            best_val = word
    return best_val




if __name__ == '__main__':
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True

    app.secret_key = 'mysecret'
    app.run(host='0.0.0.0', port=5000)
