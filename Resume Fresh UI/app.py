from flask import Flask, render_template, request, session, url_for, redirect, jsonify,make_response
import pymysql
from pyresparser import ResumeParser
from werkzeug.utils import secure_filename
from models.keras_first_go import KerasFirstGoModel
from clear_bash import clear_bash
import os
import nltk
from pydub import AudioSegment 
import speech_recognition as sr 
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.stem import PorterStemmer
import os
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
import string
punc=string.punctuation
from FacebookPostsScraper import FacebookPostsScraper as Fps
from pprint import pprint as pp
from selenium import webdriver 
from time import sleep 
from webdriver_manager.chrome import ChromeDriverManager 
from selenium.webdriver.chrome.options import Options  
from selenium.common.exceptions import NoSuchElementException  
import utils
from bs4 import BeautifulSoup
from selenium.webdriver.common.keys import Keys
import io
from bs4 import BeautifulSoup
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
#from withoutui import startcamera 
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
from voiceemotiontest import filecallingvoice
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import tensorflow as tf
from voiceemotiontest import filecallingvoice
from voiceemotiontest import loadingmodel

from filechunktextgenerate import filechunkandgenartetext

import threading
import cv2
import numpy as np
import pyaudio
import wave

CHUNK = 1024 
FORMAT = pyaudio.paInt16 #paInt8
CHANNELS = 2 
RATE = 44100 #sample rate
RECORD_SECONDS = 10
WAVE_OUTPUT_FILENAME = "nmk12.wav"





graph =tf.compat.v1.reset_default_graph()
emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
def remove_stopwords(data):
    output_array=[]
    for sentence in data:
        temp_list=[]
        for word in sentence.split():
            if word.lower() not in stop_words:
                temp_list.append(word)
        output_array.append(' '.join(temp_list))
    return output_array
def remove_punc(data):
    output_array=[]
    for sentence in data:
        temp_list=[]
        for word in nltk.word_tokenize(sentence):
            if word not in punc:
                temp_list.append(word)
        output_array.append(' '.join(temp_list))
    return output_array
from nltk.corpus import wordnet
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)
def lemmatization(data):
    output_array=[]
    for sentence in data:
        temp_list=[]
        for word in sentence.split():
            word=word.lower()
            from nltk.stem import WordNetLemmatizer
            lemma=WordNetLemmatizer()
            new_word=lemma.lemmatize(word, get_wordnet_pos(word))
            #print(new_word)
            temp_list.append(new_word)
        output_array.append(' '.join(temp_list))
    return output_array


cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

emotionlistforadding=['Angry','Disgusted','Fearful','Happy','Neutral','Sad','Surprised']
emoji_dist={0:"./emojis/angry.png",2:"./emojis/disgusted.png",2:"./emojis/fearful.png",3:"./emojis/happy.png",4:"./emojis/neutral.png",5:"./emojis/sad.png",6:"./emojis/surpriced.png"}
global last_frame1                                    
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1
show_text=[0]
global listofemotion
listofclasses=['ENFJ',
 'ENFP',
 'ENTJ',
 'ENTP',
 'ESFJ',
 'ESFP',
 'ESTP',
 'INFJ',
 'INFP',
 'INTJ',
 'INTP',
 'ISFJ',
 'ISFP',
 'ISTJ',
 'ISTP']
Corpus = pd.read_csv(r"processed_data.csv",encoding='latin-1',nrows=10,error_bad_lines=False)
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(Corpus['text_final'])


filename='naive_bayes_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))


app = Flask(__name__)
app.secret_key = 'random string'
cleaner=clear_bash()

#app.config[‘UPLOAD_FOLDER’]=
app.config['UPLOADED_FILE'] = 'static/ResumeFiles'

usr='aiinterviewer99@gmail.com'#input('Enter Email Id:')  
pwd='varshnag'#input('Enter Password:')  
usrlnk='aiinterviewer99@gmail.com'#input('Enter Email Id:')  
pwdlnk='varshnag'#input('Enter Password:')  

fbpostdata=''
linkedindata=''

#train_model()
#processed_text = first_go_model.prediction("Oracle Soap Sdlc C Engineering Opencv Architecture Android Sql Java Html Database Agile Technical")
#result = {'Job': processed_text}
#print(result)


import pyaudio
import wave
def putques(num):
    cap = cv2.VideoCapture(0)
    while True:
    # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        if not ret:
            break
        bounding_box = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            #with graph.as_default():
            
            con = dbConnection()
            cursor = con.cursor()
            query="select question from teacherquestion"
            cursor.execute(query)
            res = cursor.fetchall()
            final_result = [list(i) for i in res]
            print(list(res))
            import time
            for i in final_result:
                print(i)
                cv2.putText(frame,i[0], (x+90, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('Video', cv2.resize(frame,(1200,860),interpolation = cv2.INTER_CUBIC))
                time.sleep(10)
        if cv2.waitKey(1) & 0xFF == ord('q'):
             break
            
            

            
@app.route('/startvideo')
def startvideo():
    global listofemotion
    listofemotion=[]
    startcamera(10)
    countallemotion={}
    for ik in emotionlistforadding:
        valis=listofemotion.count(ik)
        
        countallemotion[ik]=valis
        print(countallemotion)
        #print(listofemotion)
    print(countallemotion['Angry'])
    con = dbConnection()
    cursor = con.cursor()
    cursor.execute('SELECT * FROM student WHERE name=%s',(str(session['name'])))
    print(session['name'])
    sql="INSERT INTO videoemotions (name,Angry,Disgusted,Fearful,Happy,Neutral,sad,surprise) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"
    val=((session['name']),countallemotion['Angry'],countallemotion['Disgusted'],countallemotion['Fearful'],countallemotion['Happy'],countallemotion['Neutral'],countallemotion['Sad'],countallemotion['Surprised'])
    cursor.execute(sql, val)
    con.commit()
    
    return render_template('Fetchedemotion.html', data=countallemotion, user=session['user'])
    #return 

def startcamera(num):
    global listofemotion
    listofemotion=[]
    cap = cv2.VideoCapture(0)
    while True:
    # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        if not ret:
            break
        bounding_box = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            #with graph.as_default():
            emotion_model.load_weights('emotion_model.h5')
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame,emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            listofemotion.append(emotion_dict[maxindex])
        cv2.imshow('Video', cv2.resize(frame,(1200,860),interpolation = cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
def filecallingvoice(num):
    CHUNK = 1024 
    FORMAT = pyaudio.paInt16 #paInt8
    CHANNELS = 2 
    RATE = 44100 #sample rate
    RECORD_SECONDS = 10
    WAVE_OUTPUT_FILENAME = "nmk12.wav"
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK) #buffer

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data) # 2 bytes(16 bits) per channel

    print("* done recording")
    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open("static/audio/"+WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
def getrecommendedjob(skills):
    first_go_model = KerasFirstGoModel()
    print(skills)
    processed_text = first_go_model.prediction(skills)    
    result = {'Job': processed_text}
    print(result)
    return processed_text
     
     
#Database Connection
def dbConnection():
    connection = pymysql.connect(host="localhost", user="root", password="root", database="resumeverificationtwo",use_unicode=True, charset='utf8')
    return connection







#close DB connection
def dbClose():
    dbConnection().close()
    return
@app.route('/contact.html')
def contact():
    return render_template('contact.html')

@app.route('/resumeverifier')
def resumeverifier():
    return render_template('Resumeverification.html')

@app.route('/audiovideoemotion', methods = ['GET', 'POST'])
def audiovideoemotion():
    if 'user' in session:
        if request.method == 'POST':
            global listofemotion
            link = request.form.get("link")
            print('link is',link)
            #fbposts = FB_post_fetch(fblink) 
            #fbposts=LinkedInfetchingscrapinguserdata(link)
            fbposts=[]
            #startcamera()
            #filecallingvoice()
            countallemotion={}
            t1 = threading.Thread(target=filecallingvoice, args=(68,))
            t2 = threading.Thread(target=startcamera, args=(10,))
            #t3 = threading.Thread(target=putques, args=(10,))
    
          
            # starting thread 1
            t1.start()
            # starting thread 2
            t2.start()
            
           # t3.start()
            import time
            time.sleep(14)
            
            if t1.is_alive():
                return render_template("interviewanalysis.html")
               
            else:
                countallemotion={}
                dataobt=[]
                for i in os.listdir("static/audio/"):
                    print(i)
                    listofamotionsinchunk=[]
                    textofvoiceis=[]
                    # Input audio file to be sliced 
                    audio = AudioSegment.from_wav("static/audio/"+str(i)) 
                    
                    n = len(audio) 
                  
                # Variable to count the number of sliced chunks 
                    counter = 1
                  
                # Text file to write the recognized audio 
                    fh = open("recognized.txt", "w+") 
              
            
                    interval = 5 * 1000
              
                
                    overlap = 0.00 * 1000
                  
                # Initialize start and end seconds to 0 
                    start = 0
                    end = 0
                  
                # Flag to keep track of end of file. 
                # When audio reaches its end, flag is set to 1 and we break 
                    flag = 0
                    folderchunk='audiochunk//'
                    try:
                        os.rmdir(folderchunk)
                    except:
                        if not os.path.exists(folderchunk):
                            os.mkdir(folderchunk)
            # Iterate from 0 to end of the file, 
            # with increment = interval 
                    for i in range(0, 2 * n, interval): 
                      
                    # During first iteration, 
                    # start is 0, end is the interval 
                        if i == 0: 
                            start = 0
                            end = interval 
                  
                    # All other iterations, 
                    # start is the previous end - overlap 
                    # end becomes end + interval 
                        else: 
                            start = end - overlap 
                            end = start + interval  
                  
                    # When end becomes greater than the file length, 
                    # end is set to the file length 
                    # flag is set to 1 to indicate break. 
                        if end >= n: 
                            end = n 
                            flag = 1
                  
                    # Storing audio file from the defined start to end 
                        chunk = audio[start:end] 
                  
                    # Filename / Path to store the sliced audio 
                        filename = folderchunk+'chunk'+str(counter)+'.wav'
              
                    # Store the sliced audio file to the defined path 
                        chunk.export(filename, format ="wav") 
                    # Print information about the current chunk 
                        print("Processing chunk "+str(counter)+". Start = "
                                        +str(start)+" end = "+str(end)) 
                  
                    # Increment counter for the next chunk 
                        counter = counter + 1
                        try:
                            datais=loadingmodel(filename)
                            listofamotionsinchunk.append(datais)
                        except:
                            pass
                        
                        r = sr.Recognizer()
                        with sr.AudioFile(filename) as source:
                            audio1 = r.record(source)
                        
                        try:
                            command = r.recognize_google(audio1)
                            print(command)
                            textofvoiceis.append(command)
                            fh.write(command+"\n")
                        #time.sleep(1)
                        except:
                            pass
                    
                        fh.close()
                        print(listofamotionsinchunk)
                        print(textofvoiceis)
                
                for i in range(len(listofamotionsinchunk)):
                    dataobt.append(textofvoiceis[i]+" classified as "+listofamotionsinchunk[i]+" ")
                
                for ik in emotionlistforadding:
                    valis=listofemotion.count(ik)
                    
                    countallemotion[ik]=valis
                    #print(countallemotion)
                    #print(listofemotion)
                #return render_template("interviewanalysis.html")
            return render_template('Fetchedemotion.html', data=countallemotion, user=session['user'],data1=dataobt,textofvoiceis1=textofvoiceis)
               
                   
                
                #return render_template('Fetchedemotion.html', data=countallemotion, user=session['user'])
        return render_template('home.html', user=session['user'])
    return redirect(url_for('index'))




@app.route('/interviewanalysis')
def interviewanalysis():
    return render_template('interviewanalysis.html')
@app.route('/home2')
def home2():
    return render_template('home.html')
@app.route('/index')
@app.route('/')
def index():
    return render_template('index.html')

#logout code
@app.route('/logout')
def logout():
    session.pop('user')
    return redirect(url_for('index'))
@app.route('/givetest', methods=["GET","POST"])
def givetest():
    if 'user' in session:
        con = dbConnection()
        cursor = con.cursor()
        sql="SELECT * from teacherquestion"
        cursor.execute(sql)
        res = cursor.fetchall()
        #print(type(res))
        cursor.execute('SELECT * FROM student WHERE name=%s',(str(session['name'])))
        cursor.execute('SELECT * FROM student WHERE id=%s',(str(session['id'])))
        if request.method == "POST":
            for i in range(1,6):
                answer=request.form.get("description"+str(i))
                sql = "INSERT INTO answers (sid ,answers) VALUES (%s,%s)"
                val = ((session['id']),answer)
                cursor.execute(sql, val)
                con.commit()
            cursor.execute("select answers from answers where sid="+str(session['id']))
            stud=cursor.fetchall()
            cursor.execute('SELECT answer from teacherquestion')
            teach=cursor.fetchall()
            stud1=[]
            for l in stud:
                stud1.append(l[0])
            teach1=[]
            for k in teach:
                teach1.append(k[0])
                
            print(teach1)
            print(stud1)
            output1=remove_stopwords(teach1)
            output2=remove_stopwords(stud1)
            output3=remove_punc(output1)
            output4=remove_punc(output2) 
            output5=lemmatization(output3)
            output6=lemmatization(output4)
            marks=[]
            finalmarks=[]
            print(output3)
            print(output4)
            print(output5)
            print(output6)
            for i in range(len(teach)):
                m=output5[i]
                n=output6[i]
                print(m)
                print(n)
                from fuzzywuzzy import fuzz
                c=fuzz.ratio(output5[i],output6[i])
                #d=fuzz.token_set_ratio(output6[i],output5[i])
                print(c)
                #print(d)
                a=c/10
                print(a)
                finalmarks.append(a)
            finalmarks
            a=finalmarks[0]
            b=finalmarks[1]
            c=finalmarks[2]
            d=finalmarks[3]
            e=finalmarks[4]
            for i in finalmarks:
                a=i
            total = 0
            for ele in range(0, len(finalmarks)):
                print(ele)
                total = total + finalmarks[ele]
            import math
            print(total)
            ma=math.ceil(total)
            print(ma)
            con = dbConnection()
            cursor = con.cursor()
            sql="INSERT INTO marks (id,sid,q1,q2,q3,q4,q5,marks) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"
            val=((session['id']),(session['name']),a,b,c,d,e,ma)
            cursor.execute(sql, val)
            con.commit()
            print("Print Marks")
            con = dbConnection()
            cursor = con.cursor() 
            cursor.execute("select * from marks WHERE id=%s",(str(session['id'])))
            result = cursor.fetchall() #data from database 
           
            
            cv2.destroyAllWindows()
            con = dbConnection()
            cursor = con.cursor() 
            cursor.execute("select * from videoemotions WHERE name=%s",(str(session['name'])))
            result2=cursor.fetchall()
            print(result2)
        
            return render_template("result.html", result=result,result2=result2)
            
           # return render_template('home.html',name=session['name'])
            #return render_template('home.html')
        return render_template('questionans.html',res=res)
    return render_template('questionans.html')


@app.route('/register', methods=["GET","POST"])
def register():
    if request.method == "POST":
        try:
            status=""
            name = request.form.get("name")
            address = request.form.get("address")
            mailid = request.form.get("email")
            mobile = request.form.get("mobno")
            pass1 = request.form.get("password2")
            con = dbConnection()
            cursor = con.cursor()
            cursor.execute('SELECT * FROM userdetails WHERE emailid = %s', (mailid))
            res = cursor.fetchone()
            #res = 0
            if not res:
                sql = "INSERT INTO userdetails (name, address, emailid, mobileno, password) VALUES (%s, %s, %s, %s, %s)"
                val = (name, address, mailid, mobile, pass1)
                print(sql," ",val)
                cursor.execute(sql, val)
                con.commit()
                status= "success"
                return redirect(url_for('login'))
            else:
                status = "Already available"
            #return status
            return redirect(url_for('index'))
        except Exception as e:
            print(e)
            print("Exception occured at user registration")
            return redirect(url_for('index'))
        finally:
            dbClose()
    return render_template('register.html')

@app.route('/login', methods=["GET","POST"])
def login():
    msg = ''
    if request.method == "POST":
        session.pop('user',None)
        mailid = request.form.get("email")
        password = request.form.get("password2")
        #print(mobno+password)
        con = dbConnection()
        cursor = con.cursor()
        result_count = cursor.execute('SELECT * FROM userdetails WHERE emailid = %s AND password = %s', (mailid, password))
        #a= 'SELECT * FROM userdetails WHERE mobile ='+mobno+'  AND password = '+ password
        #print(a)
        #result_count=cursor.execute(a)
        # result = cursor.fetchone()
        res=cursor.fetchone()
        if result_count>0:
            print(result_count)
            session['name']=res[1]
            session['id']=res[0]
            session['user'] = mailid
            return redirect(url_for('home2'))
        else:
            print(result_count)
            msg = 'Incorrect username/password!'
            return msg
    
    return render_template('login.html')

#login code
@app.route('/login1', methods=["GET","POST"])
def login1():
    msg = ''
    if request.method == "POST":
        session.pop('user',None)
        mailid = request.form.get("mailid")
        password = request.form.get("pas")
        #print(mobno+password)
        con = dbConnection()
        cursor = con.cursor()
        result_count = cursor.execute('SELECT * FROM userdetails WHERE emailid = %s AND password = %s', (mailid, password))
        #a= 'SELECT * FROM userdetails WHERE mobile ='+mobno+'  AND password = '+ password
        #print(a)
        #result_count=cursor.execute(a)
        # result = cursor.fetchone()
        if result_count>0:
            print(result_count)
            session['user'] = mailid
            return redirect(url_for('home'))
        else:
            print(result_count)
            msg = 'Incorrect username/password!'
            return msg
        #dbClose()
    return redirect(url_for('index'))


#user register code
@app.route('/userRegister', methods=["GET","POST"])
def userRegister():
    if request.method == "POST":
        try:
            status=""
            name = request.form.get("name")
            address = request.form.get("address")
            mailid = request.form.get("mailid")
            mobile = request.form.get("mobile")
            pass1 = request.form.get("pass1")
            con = dbConnection()
            cursor = con.cursor()
            cursor.execute('SELECT * FROM userdetails WHERE emailid = %s', (mailid))
            res = cursor.fetchone()
            #res = 0
            if not res:
                sql = "INSERT INTO userdetails (name, address, emailid, mobileno, password) VALUES (%s, %s, %s, %s, %s)"
                val = (name, address, mailid, mobile, pass1)
                print(sql," ",val)
                cursor.execute(sql, val)
                con.commit()
                status= "success"
                return redirect(url_for('index'))
            else:
                status = "Already available"
            #return status
            return redirect(url_for('index'))
        except:
            print("Exception occured at user registration")
            return redirect(url_for('index'))
        finally:
            dbClose()
    return redirect(url_for('index'))


@app.route('/home')
def home():
    if 'user' in session:

        return render_template('home.html', user=session['user'])
    return redirect(url_for('index'))

#import os

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    global fbpostdata,linkedindata
    if 'user' in session:
        if request.method == 'POST':
            f = request.files['file']
            print(f)
            filename = secure_filename(f.filename)
            f.save(os.path.join(app.config['UPLOADED_FILE'], filename))

            filename = os.path.abspath(app.config['UPLOADED_FILE']+"//"+filename)#os.path.abspath(filename)
            
            filename = filename#'ningesh.pdf'
 #           print("=======")
 #           print(filename)
 #           print("=======")
            listofcontent = ['pdf', 'docx']
            splitfilename = filename.split(".")
            if splitfilename[1] in listofcontent:
                data = ResumeParser(filename).get_extracted_data()
                print(data)
                #print(type(['Soap', 'Agile', 'Sdlc', 'Java', 'Sql', 'C', 'Android', 'Engineering', 'Html', 'Architecture', 'Opencv', 'Database', 'Oracle', 'Technical']))
                                
                skillset=data['skills']
                print(skillset)
                dataop=''
                for ik in skillset:
                    dataop=dataop+ik+" "
                
                print('=======')
                print(dataop)
                #global first_go_model
                #processed_text1 = first_go_model.prediction("Oracle Soap Sdlc C Engineering Opencv Architecture Android Sql Java Html Database Agile Technical")
                #processed_text1 = first_go_model.prediction(dataop)
                
                processed_text1 = getrecommendedjob(dataop)
                result1 = {'Job type suited for this ': processed_text1}
                
                #result1 = {'Job type suited for this ': processed_text1}
                print(result1)
                
                processed_text12 = getrecommendedjob(linkedindata)
                resultlinkedin = {'Job type suited for this from linked in post': processed_text12}
                
                #result1 = {'Job type suited for this ': processed_text1}
                print('resultlinkedin',resultlinkedin)
                predictions_RF = loaded_model.predict(Tfidf_vect.transform([fbpostdata]))
                #print(listofclasses[predictions_RF[0]-1])
                #predictions_RF = loaded_model.predict(Tfidf_vect.transform([fbpostdata]))
                processed_text11 =listofclasses[predictions_RF[0]-1]#getpersonalityprediction(fbpostdata)
                resultfacebookpersonality = {'Personality prediction from post ': processed_text11}
                
                #result1 = {'Job type suited for this ': processed_text1}
                print('resultfacebook',resultfacebookpersonality)
                
                
                
                
                return render_template('recommend.html', user=session['user'], data=data, job=processed_text1,linkedininfo=processed_text12,fbpersonality=processed_text11)
            else:
                print('Not able to scan data please provide pdf format')


            #return 'file uploaded successfully'
        return render_template('home.html', user=session['user'])
    return redirect(url_for('index'))

    

@app.route('/FBLinkuploader', methods = ['GET', 'POST'])
def FBLinkuploader():
    if 'user' in session:
        if request.method == 'POST':
            fblink = request.form.get("fblink")
            #fbposts = FB_post_fetch(fblink) 
            fbposts=fetchingscrapinguserdata(fblink)
            
            return render_template('FetchedPost.html', data=fbposts, user=session['user'])
        return render_template('home.html', user=session['user'])
    return redirect(url_for('index'))

@app.route('/LinkedinLinkuploader', methods = ['GET', 'POST'])
def LinkedinLinkuploader():
    if 'user' in session:
        if request.method == 'POST':
            link = request.form.get("link")
            print('link is',link)
            #fbposts = FB_post_fetch(fblink) 
            fbposts=LinkedInfetchingscrapinguserdata(link)
            
            return render_template('FetchedPost.html', data=fbposts, user=session['user'])
        return render_template('home.html', user=session['user'])
    return redirect(url_for('index'))

@app.route('/clickvideoemotion', methods = ['GET', 'POST'])
def clickvideoemotion():
    if 'user' in session:
        if request.method == 'POST':
            global listofemotion
            link = request.form.get("link")
            print('link is',link)
            #fbposts = FB_post_fetch(fblink) 
            #fbposts=LinkedInfetchingscrapinguserdata(link)
            fbposts=[]
            startcamera()
            countallemotion={}
            for ik in emotionlistforadding:
                valis=listofemotion.count(ik)
                
                countallemotion[ik]=valis
                print(countallemotion)
                print(listofemotion)
            return render_template('Fetchedemotion.html', data=countallemotion, user=session['user'])
        return render_template('home.html', user=session['user'])
    return redirect(url_for('index'))

@app.route('/startspeechemotion', methods = ['GET', 'POST'])
def startspeechemotion():
    if 'user' in session:
        if request.method == 'POST':
            global listofemotion
            listofemotion=[]
            link = request.form.get("link")
            print('link is',link)
            #fbposts = FB_post_fetch(fblink) 
            #fbposts=LinkedInfetchingscrapinguserdata(link)
            fbposts=[]
            data1,textofvoiceis=filecallingvoice()
            #data1=loadingmodel()
            #startcamera()
            countallemotion={}
            dataobt=[]
            for i in range(len(data1)):
                dataobt.append(textofvoiceis[i]+" classified as "+data1[i]+" ")
            
            for ik in emotionlistforadding:
                valis=listofemotion.count(ik)
                
                countallemotion[ik]=valis
                print(countallemotion)
                print(listofemotion)
            return render_template('SpeechFetchedemotion.html', data=dataobt,textofvoiceis1=textofvoiceis, user=session['user'])
        return render_template('home.html', user=session['user'])
    return redirect(url_for('index'))

def LinkedInfetchingscrapinguserdata(links):
    global usrlnk
    global pwdlnk,linkedindata
  
    driver = webdriver.Chrome(ChromeDriverManager().install()) 
    driver.get('https://www.linkedin.com') 
    print ("Opened linkedin") 
    sleep(1) 

    username_box = driver.find_element_by_id('session_key') 
    username_box.send_keys(usrlnk) 
    print ("Email Id entered") 
    sleep(1) 
  
    password_box = driver.find_element_by_id('session_password') 
    password_box.send_keys(pwdlnk) 
    print ("Password entered") 
  
#login_box = driver.find_element_by_id('loginbutton') 
#login_box.click() 

    try:
            # clicking on login button
            driver.find_element_by_class_name("sign-in-form__submit-button").click()
    except NoSuchElementException:
            # Facebook new design
            driver.find_element_by_name("Sign in").click()
  
    print ("Done") 

    sleep(10) 
    #links='https://www.linkedin.com/in/dipti-mhatre-2a04b452/'
    driver.get(links)
    sleep(5) 

    try:
            # clicking on login button
            driver.find_element_by_class_name("pv-skills-section__chevron-icon").click()
    except NoSuchElementException:
            # Facebook new design
            pass



    html = driver.find_element_by_tag_name('html')  
    SCROLL_PAUSE_TIME = 1

# Get scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")

    ik=0
    while ik<5:
        ik+=1
    
        html.send_keys(Keys.END)
        sleep(1)
    
    sleep(10)
    page_text = (driver.page_source).encode('utf-8')



#soup = BeautifulSoup(page_text)

#parse_data = soup.get_text()
#soup = BeautifulSoup(page_text, 'html.parser')
    
    with io.open("outputlinkedin.html", "w", encoding="utf-8") as f:
        f.write(str(page_text))
    

#print(soup)
#reviews_selector = soup.find_all('div', class_='_5msj')
#print(reviews_selector)
#row = soup.find('div._5msj') 
#print(row)


    person = {}



    print('data is ',person)
#sleep(5)     
#sleep(5) 
#input('Press anything to quit') 
    driver.quit() 
    print("Finished") 
    with open("outputlinkedin.html") as html_file:
        html = html_file.read()
    
## creating a BeautifulSoup object
    soup = BeautifulSoup(html, "html.parser")
#soup = BeautifulSoup(htmltxt, 'lxml')
    p_tags = soup.find_all("h3")  
    abbr_tags = soup.find_all("h4") 
    
    listofalltext=[]
    listofalldate=[]
    alldata=[]
#print(soup.find_all('p'))
    for tag in p_tags:
        listofalltext.append(tag.text)
        #print(tag.text)
    for tag in abbr_tags:
        listofalldate.append(tag.text)
    

    for i in range(len(listofalldate)):  
        linkedindata=linkedindata+listofalltext[i]+' '
        alldata.append(listofalltext[i]+"@"+listofalldate[i])
    #print(listofall[i]+"@"+listofalldate[i])

    print(alldata)
    return alldata

def FB_post_fetch(link):
    # Enter your Facebook email and password
    email = 'YOUR_EMAIL'
    password = 'YOUR_PASSWORD'
    # Instantiate an object
    fps = Fps(email, password, post_url_text='Full Story')
    # Example with single profile
    #single_profile = 'https://www.facebook.com/BillGates'
    data = fps.get_posts_from_profile(link)
    pp(data)
    return data

def fetchingscrapinguserdata(link):
    global usr
    global  pwd,fbpostdata
    driver = webdriver.Chrome(ChromeDriverManager().install()) 
    driver.get('https://m.facebook.com/') 
    print ("Opened facebook") 
    sleep(1) 

    username_box = driver.find_element_by_id('m_login_email') 
    username_box.send_keys(usr) 
    print ("Email Id entered") 
    sleep(1) 
  
    password_box = driver.find_element_by_id('m_login_password') 
    password_box.send_keys(pwd) 
    print ("Password entered") 
  
#login_box = driver.find_element_by_id('loginbutton') 
#login_box.click() 

    try:
            # clicking on login button
            driver.find_element_by_id("loginbutton").click()
    except NoSuchElementException:
            # Facebook new design
            driver.find_element_by_name("login").click()
  
    print ("Done") 

    sleep(15) 
    #link='https://m.facebook.com/ningeshkumar.kharatmol/'
    driver.get(link)
    sleep(5) 
    html = driver.find_element_by_tag_name('html')  
    SCROLL_PAUSE_TIME = 1

# Get scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")

    ik=0
    while ik<25:
        ik+=1
    
        html.send_keys(Keys.END)
        sleep(1)
    
    sleep(10)
    page_text = (driver.page_source).encode('utf-8')
#soup = BeautifulSoup(page_text)

#parse_data = soup.get_text()
#soup = BeautifulSoup(page_text, 'html.parser')

    with io.open("output1.html", "w", encoding="utf-8") as f:
        f.write(str(page_text))
    
    
    

#print(soup)
#reviews_selector = soup.find_all('div', class_='_5msj')
#print(reviews_selector)
#row = soup.find('div._5msj') 
#print(row)


    person = {}



    print('data is ',person)
#sleep(5)     
#sleep(5) 
#input('Press anything to quit') 
    driver.quit() 
    print("Finished") 
    
    with open("output1.html") as html_file:
        html = html_file.read()
    
## creating a BeautifulSoup object
    soup = BeautifulSoup(html, "html.parser")
#soup = BeautifulSoup(htmltxt, 'lxml')
    p_tags = soup.find_all("p")  
    abbr_tags = soup.find_all("abbr") 
    listofalltext=[]
    listofalldate=[]
    alldata=[]
#print(soup.find_all('p'))
    for tag in p_tags:
        listofalltext.append(tag.text)
        #print(tag.text)
    for tag in abbr_tags:
        listofalldate.append(tag.text)
    

    for i in range(len(listofalltext)):  
        #alldata.append(listofalltext[i])
        fbpostdata=fbpostdata+listofalltext[i]+' '
        alldata.append(listofalltext[i]+"@"+listofalldate[i])
    #print(listofall[i]+"@"+listofalldate[i])

    print(alldata)
    return alldata

if __name__ == '__main__':
    #app.run(debug="True")
    app.run('0.0.0.0')
   