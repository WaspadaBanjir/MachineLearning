from flask import Flask
from flask_restful import Resource, Api
from flask_restful import reqparse
from flask.ext.mysql import MySQL
import pandas as pd
import numpy as np
from sklearn import svm, datasets
import matplotlib.pyplot as plt

mysql = MySQL()
app = Flask(__name__)

# MySQL configurations
app.config['MYSQL_DATABASE_USER'] = 'root'
# app.config['MYSQL_DATABASE_PASSWORD'] = 'jay'
app.config['MYSQL_DATABASE_DB'] = 'Siaabdb'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'


mysql.init_app(app)

api = Api(app)

class TrainData(Resource):
    def post(self):
        try:
            conn = mysql.connect()
            cursor = conn.cursor()
            cursor.callproc('sp_GetTrainingData')
            data = cursor.fetchall()

            classifiers = []
            features = []
            for i in data:
                if i[2] > 0:
                    classifiers.append(1)
                else:
                    classifiers.append(0)
                list1 = list(i)
                del list1[-1]
                features.append(list1)

            X = np.array(features)
            y = np.array(classifiers)

            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            h = (x_max / x_min)/100
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
            # X_plot = np.c_[xx.ravel(), yy.ravel()]

            # Create the SVC model object
            C = 1.0 # SVM regularization parameter
            svc = svm.SVC(kernel='rbf', C=C, decision_function_shape='ovr').fit(X, y)

            from sklearn.externals import joblib
            joblib.dump(svc, 'filename.pkl') 
            # Z = svc.predict(X_plot)
            # Z = Z.reshape(xx.shape)
            print("File saved")

        except Exception as e:
            return {'error': str(e)}


class GetStatus(Resource):
    def post(self):
        try: 
            # Parse the arguments
            parser = reqparse.RequestParser()
            parser.add_argument('kelurahan', type=str)
            args = parser.parse_args()

            _kelurahan = args['kelurahan']
            _kelurahan = "Manggarai"

            conn = mysql.connect()
            cursor = conn.cursor()
            cursor.callproc('sp_GetStatus',[_kelurahan])
            data = cursor.fetchall()

            

            return {'StatusCode':'200','Items':data}

        except Exception as e:
            return {'error': str(e)}

api.add_resource(TrainData, '/TrainData')
api.add_resource(GetStatus, '/GetStatus')

if __name__ == '__main__':
    app.run(debug=True)