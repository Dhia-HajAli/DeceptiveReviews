import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

app = Flask(__name__)
model = pickle.load(open('rf.h5', 'rb'))
vectorizer = pickle.load(open('vectorizer.h5', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    data = request.form.get('text')
    # Make prediction
    review = re.sub('[^a-zA-Z]',' ',data)
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    review=[review]
    review_test = vectorizer.transform(review).toarray()

    prediction = model.predict_proba(review_test).tolist()
    max_value = max(prediction[0])
    ind_max = prediction[0].index(max_value)

    return render_template('index.html', pred=max_value, ind=ind_max)
@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    review = re.sub('[^a-zA-Z]',' ',str(data))
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    review=[review]
    review_test = vectorizer.transform(review).toarray()

    prediction = model.predict_proba(review_test).tolist()
    max_value = max(prediction[0])
    ind_max = prediction[0].index(max_value)
    
    if (ind_max==1):
        return jsonify("This review is a ",max_value," truthfull ",1-max_value, " Deceptive" )
    else:
        return jsonify(max_value, ind_max)

if __name__ == "__main__":
    app.run(debug=True)