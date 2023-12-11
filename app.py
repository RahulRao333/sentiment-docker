import pickle
from flask import Flask,render_template,request,jsonify


import joblib
#import mysql.connector
app = Flask(__name__,static_url_path='/static',template_folder='templates')



# Configure MySQL

db_config = {
    'host': 'mysql',
    'user': 'root',
    'password': '1234',
    'database': 'sentiment',
    # Use appropriate authentication plugin
}
# Initialize MySQL
#conn = mysql.connector.connect(**db_config)

# Create Cursor Object
#cur = conn.cursor()
modelrf=pickle.load(open('rfmodel.pkl','rb'))
modelvect=joblib.load('vectorizer.pkl')



@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':




        message=request.form['inputData']
        data=[message]



        vect= modelvect.transform(data)



        myprediction=modelrf.predict(vect)
        output=myprediction


        insertquery='insert into predictions (review,sentiment) values (%s,%s)'
        #cur.execute(insertquery,(message,output))
        #mysql.connection.commit()
        #cur.close()
        if output == [1]:
            print(output)
            return render_template('index.html', prediction_text='Its a Positive Review')
        if output == [0]:
            print(output)
            return render_template('index.html', prediction_text='Its a Negative Review')





if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5000)