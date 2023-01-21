from flask import Flask,render_template,flash, redirect,url_for,session,logging,request,Response,send_file
from flask_login import LoginManager, UserMixin,login_user,login_required,logout_user,current_user
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from io import BytesIO
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
import os
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import cv2

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from tensorflow import keras
from keras.applications.resnet import preprocess_input
from keras.models import load_model
# from keras.preprocessing import image

from keras.utils import load_img, img_to_array



MODEL_PATH ='D:\My_Projects\chili_plant\model_inception.h5'

# Load your trained model
model = load_model(MODEL_PATH)


def model_predict(img_path, model):
    print(img_path)
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x=x/255
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="Healthy"
    elif preds==1:
        preds="Leaf Curl Disease"
    elif preds==2:
        preds="Leaf Spot Disease"
    elif preds==3:
        preds="Whitefly Disease"
    elif preds==4:
        preds="Yellowish Disease"
    else:
        preds="Not Matched"
    return preds

app = Flask(__name__,template_folder='template')
app.secret_key = 'hello'

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
db.init_app(app)

login_manger =LoginManager()
login_manger.init_app(app)

class Users(UserMixin,db.Model):
    __tablename__= 'users'
    id = db.Column(db.Integer, primary_key=True,autoincrement=True)
    name = db.Column(db.String(120))
    email = db.Column(db.String(80),unique=True)
    password = db.Column(db.String(80))

class Reports(UserMixin,db.Model):
    __tablename__= 'reports'
    report_id = db.Column(db.Integer, primary_key=True,autoincrement=True)
    # week = db.Column(db.String(120),unique=True)
    # file_path = db.Column(db.String(1000))
    current_date = db.Column(db.Date)
    image = db.Column(db.LargeBinary)
    pred_val = db.Column(db.String(100))
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))

@login_manger.user_loader
def load_user(user_id):
    return Users.query.get(int(user_id))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login',methods=['GET','POST'])
def login():
    if request.method == "POST":
        u_email = request.form["u_email"]
        u_pwd = request.form["u_pwd"]
        login = Users.query.filter_by(email = u_email, password=u_pwd).first()
        if login is not None:
            login_user(login)
            return redirect('/home')
        else:
            flash('Username or password is wrong')
            return redirect('/login')
    return render_template("login.html")

@app.route('/register',methods=['GET','POST'])
def register():
    if request.method == "POST":
        _uname = request.form["uname"]
        _email = request.form["email"]
        _passw = request.form["pwd"]
        _cpassw = request.form["cpwd"]
        if _passw == _cpassw:
            register = Users(name=_uname, email=_email, password=_passw)
            db.session.add(register)
            db.session.commit()
            return redirect('/login')
        else:
            flash('Password and Confirm Password does not match')
            return redirect('/register')
    return render_template("register.html")

@app.route('/home')
@login_required
def home():
    return render_template('home.html',users=current_user)



@app.route('/predict', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'upload', secure_filename(f.filename))
        f.save(file_path)
        preds = model_predict(file_path, model)
        result = preds
        _user_id = current_user.id
        today = datetime.today()
        to_date = today.strftime("%Y-%m-%d")
        t_date = datetime.strptime(to_date, '%Y-%m-%d').date()
        convert_to_png(file_path)
        with open(file_path, "rb") as fi:
            binary_data = fi.read()
            report = Reports( current_date=t_date,image=binary_data, pred_val=result, user_id=_user_id)
            db.session.add(report)
            db.session.commit()

        return result
    return render_template('home.html',result=result)



def convert_to_png(image_path):
    # Open the image file
    img = cv2.imread(image_path)
    # Save the image as a PNG file
    cv2.imwrite(image_path.replace(".jpg", ".png"), img)

@app.route('/image/<int:image_id>')
def image(image_id):
    reports = Reports.query.get(image_id)
    return send_file(BytesIO(reports.image), mimetype='image/png')

@app.route('/report',methods=['GET','POST'])
@login_required
def report():
    if request.method == "GET":
        _user_id = current_user.id
        report = Reports.query.filter_by(user_id=_user_id).all()
    return render_template("report.html",report=report,users=current_user)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)
