from flask import Flask, render_template, request, jsonify
import pickle 
import pandas as np
from fastai.text.all import *
from transformers import *
from blurr.data.all import *
from blurr.modeling.all import *
import pathlib
inf_learn = load_learner(fname='./best.pkl')

def Summarise(input_data):
    def check(x):
        kickoff = [item.replace("\xa0", "") for item in x]
        kickoff1 = [item.replace("A", "") for item in kickoff ]
        kickoff2 = [item.replace("\x80", "") for item in kickoff1]
        kickoff3 = [item.replace("\x98", "") for item in kickoff2]
        kickoff4 = [item.replace("Â", "") for item in kickoff3]
        kickoff5 = [item.replace("º", "") for item in kickoff4]
    
        return kickoff5
    return check(inf_learn.blurr_generate(input_data))
 
def main():
    Article = str(input("Enter the article to be summarized: "))

    Summary = Summarise(Article)
    print(Summary)

# if __name__ == '__main__':
#     main()

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('./index.html')

@app.route("/summarise", methods=('POST', 'GET'))    
def sm():
    args = request.args
    print(args.get('text'))
    Summary = Summarise(str(args.get('text')))
    print(Summary[0])
    jsonResp = {'summary': Summary[0]}
    print(jsonResp)
    # return 'ok'
    return jsonify(jsonResp)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)