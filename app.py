from flask import Flask, make_response, request, render_template
import pandas as pd
import keras
import numpy as np
model = keras.models.load_model('/actionnormal.h5')
app = Flask(__name__)

@app.route('/')
def form(prediction_text = 'your predicted activity will be shown here in series'):
    return """
        <html>
            <body>
                <h1>Human Activity Prediction</h1>
                <h3>Kindly upload Device Motion recording using SensingKit or Crowd Sense at 50hz for prediction</h3>
                <form action="/predict" method="post" enctype="multipart/form-data">
                    <input type="file" name="data_file" />
                    <input type="submit" />
                </form>
                <br><br><h3>{prediction_text}</h3><br>
                </body>
            </body>
        </html>
    """.format(prediction_text=prediction_text)


@app.route('/predict', methods=["POST",'GET'])
def predict():
  try:

    f = request.files['data_file']
    if not f:
        return "No file"
    motiondata = pd.read_csv(f)
    seriesin = np.array(motiondata[['attitude.roll'	,'attitude.pitch'	,'attitude.yaw',	'gravity.x'	,'gravity.y'	,'gravity.z'	,'rotationRate.x',	'rotationRate.y',	'rotationRate.z'	,'userAcceleration.x'	,'userAcceleration.y',	'userAcceleration.z']])

    rate = 250
    reminder  = (len(seriesin))%rate
    serieslenthtotake = len(seriesin) - reminder
    seriesinput = np.array([seriesin[-serieslenthtotake:][n:n+rate] for n in range(0, len(seriesin[-serieslenthtotake:]), rate)])
    predictions = model.predict(seriesinput)
    max_predictions = np.argmax(predictions, axis=1)
    my_dict ={1:'downstairs', 4:'jogging', 0:'sitting', 3:'standing', 2:'upstairs', 5:'waling'}
    activity_predition = [my_dict[x] for x in max_predictions]

    return (form(prediction_text = activity_predition))
  except:
    return (form(prediction_text = 'please check csv file again and try again'))  

if __name__ == "__main__":
	app.run()

