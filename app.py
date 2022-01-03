from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import sys

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if random_forest:
        try:
            json_ = request.json
            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)

            prediction = list(random_forest.predict(query))

            return jsonify({'prediction': str(prediction)})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except:
        port = 12345 

    random_forest = joblib.load("model.pkl")
    print ('Model loaded')
    model_columns = joblib.load("model_columns.pkl") 
    print ('Model columns loaded')

    app.run(port=port)
    
#This API is not available on the localhost, it requires the Postman software in order to function 