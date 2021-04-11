from flask import Flask, request, send_file, render_template
from flask_uploads import UploadSet, configure_uploads, IMAGES
import logging
import os

app = Flask(__name__, static_folder='static/')

UPLOAD_FOLDER = 'data'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.config['UPLOADED_PHOTOS_DEST'] = UPLOAD_FOLDER
photos = UploadSet('photos', ['xlsx'])
configure_uploads(app, photos)



logging.basicConfig(level=logging.INFO)

@app.route("/")
def root():
    return render_template("final.html")

@app.route("/predict",methods = ["POST", "GET"])
def predict_api():
    if request.method == 'POST' and ('input_csv' in request.files):
        input_csv = request.files.get("input_csv")
        in_csv_name = save_file(input_csv)
        filename = predict(in_csv_name)
        if not filename:
            return {"success":False, "message":"Something went wrong"}, 500
        else:
            return send_file(
                filename,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                as_attachment=True)
    else:
        return {"success":False, "message":"Bad input"}, 400
        
@app.route("/download",methods = ["GET"])
def download_xlsx_sheet():
    payload = request.args
    filename = payload.get("filename")
    if not filename or not os.path.isfile(filename):
        return {"success":False, "message":"Bad input"}, 400
    return send_file(
                filename,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                as_attachment=True)

from model import predict
from utils import save_file