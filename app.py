from flask import Flask,render_template,url_for,request,jsonify
import pickle
import joblib

filename = 'pickle.pkl'
clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('tranform.pkl','rb'))
app = Flask(__name__)

@app.route('/', methods=['GET'])

def home():
	msg = request.args['msg']
	data = [msg]
	vect = cv.transform(data).toarray()
	prediction = clf.predict(vect)
	result = prediction[0]
	if result==1:
		return "hey there this is spam"
	else:
		return "hey this is Not Spam"


if __name__ == '__main__':
	app.run(debug=True)