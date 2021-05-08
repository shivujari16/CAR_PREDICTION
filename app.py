import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)
model = joblib.load(open('car_prediction_final1.sav', 'rb'))

@app.route('/')
def home():
    return render_template('shivam.html')

list1 = ['alfa', 'audi', 'bmw', 'chevrolet', 'dodge', 'honda', 'isuzu',
       'jaguar', 'mazda', 'buick', 'mercury', 'mitsubishi',
       'nissan', 'peugeot', 'plymouth', 'porsche', 'renault',
       'saab', 'subaru', 'toyota', 'volkswagen','volvo']

list2 = ['rwd', 'fwd', '4wd']

list3 = ['front', 'rear']

@app.route('/predict',methods=['POST'])
def predict():
    
    searchValue = request.form['CarName']
    x = [i for i,n in enumerate(list1) if n == searchValue]
    for a in x:
            y=a

    searchValue1 = request.form['drivewheel']
    x = [i for i,n in enumerate(list2) if n == searchValue1]
    for a in x:
         y1=a
         
    searchValue2 = request.form['enginelocation']
    x = [i for i,n in enumerate(list3) if n == searchValue2]
    for a in x:
         y2=a
         
         

                                   
    input_features = [y,y1,y2,request.form['wheelbase'],request.form['carlength'],request.form['carwidth'],
                      request.form['curbweight'],request.form['cylindernumber'],request.form['enginesize'],request.form['boreratio'],
                      request.form['horsepower'],request.form['citympg'],request.form['highwaympg']]                
    features_value = [np.array(input_features)]
    
    features_name = ['CarName', 'drivewheel','enginelocation', 'wheelbase', 'carlength', 'carwidth',
        'curbweight', 'cylindernumber', 'enginesize', 'boreratio', 'horsepower', 'citympg', 'highwaympg']
   
    print(features_value)
    
    df = pd.DataFrame(features_value, columns=features_name)
    output = (model.predict(df).round(2))
    
    ot = str(output).split('.')
    
    out = ot[0]+ot[1]
        

    return render_template('shivam.html', prediction_text='Price is {} Rupees'.format(out))

if __name__ == "__main__":
    app.run()