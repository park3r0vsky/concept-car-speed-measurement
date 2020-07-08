from datetime import datetime
from os import listdir
from os.path import getctime
from time import sleep
from subprocess import call
from IPython import get_ipython
try:
    import Image
except ImportError:
    from PIL import Image
import cv2
import pytesseract
import numpy as np


distance_meters = 1300
velocity_limit_kmh = 140

velocity_limit_ms = velocity_limit_kmh *(1000/3600) 
time_limit = distance_meters/velocity_limit_ms


#Path to pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\park3r\\Anaconda3\\pkgs\\tesseract-3.05.01-vc14_10\\Library\\bin\\tesseract.exe'

pathRawImg='C:\\Users\\park3r\\Anaconda3\\yolo\\pytorch-yolo-v3\\img_auto'
pathDetImg='C:\\Users\\park3r\\Anaconda3\\yolo\\pytorch-yolo-v3\\detz_auto'

#Run detection process 
pathDetScript = r'C:\\Users\\park3r\\Anaconda3\\yolo'
cmdLine = 'activate_yolo_auto.bat'
rc = call("start cmd /K " + cmdLine, cwd=pathDetScript, shell=True)



def checkDetection():
    while True:
        try:
            process_check =  open('%s//det_%s'% (pathDetImg,listdir(pathRawImg)[-1]), 'r')
            process_check.close()
            break
        except:
            get_ipython().magic('clear')
            print("Detecting.", end="", flush=True)
            sleep(1) 
            get_ipython().magic('clear')
            print("Detecting..", end="", flush=True)
            sleep(1) 
            get_ipython().magic('clear')
            print("Detecting...", end="", flush=True)
            sleep(1) 
            continue
    print('Done')
    print('Reading done')
    print('Showing results')


#Optical Character Recognition
def ocr(img):
    return pytesseract.image_to_string(img, lang='leu')



def getLenOfLastWord(s):
    words = s.split()
    if len(words) == 0 or words[0] == words[-1]:
        return 0
    return len(words[-1])   



def drawRectInfobox(image, color):
    cover = image.copy()
    x, y, w, h = 0, 2000, 3504, 336  
    alpha = 0.5  
    if color == 'red':
        cv2.rectangle(cover, (x, y), (x+w, y+h), (0, 0, 255), -1)
        return cv2.addWeighted(cover, alpha, image, 1 - alpha, 0) 
    elif color == 'green':
        cv2.rectangle(cover, (x, y), (x+w, y+h), (0, 200, 0), -1)
        return cv2.addWeighted(cover, alpha, image, 1 - alpha, 0)   
    else:
        pass

    
    
def putTextInfobox(img, tsec, plate_number, filename, date, clocktime):

    vms = distance_meters/tsec
    vkh = round(vms * 3.6)
    v_text = 'Pomiar: ' + str(vkh) + ' km/h'
    d_text = 'Data: ' + str(date) 
    h_text = 'Godzina: ' + str(clocktime)
    vt_text = 'Limit: ' + str(velocity_limit_kmh) + ' km/h'
    pl_text = 'Numer: ' + str(plate_number)
    
    cv2.putText(img, v_text, (600, 2250), cv2.FONT_HERSHEY_PLAIN, 15, [255,255,255], 12)
    cv2.putText(img, filename, (150, 150), cv2.FONT_HERSHEY_PLAIN, 7, [255,255,255], 12)
    cv2.putText(img, d_text, (2300, 150), cv2.FONT_HERSHEY_PLAIN, 7, [255,255,255], 12)
    cv2.putText(img, h_text, (2300, 250), cv2.FONT_HERSHEY_PLAIN, 7, [255,255,255], 12)
    cv2.putText(img, vt_text, (2300, 350), cv2.FONT_HERSHEY_PLAIN, 7, [255,255,255], 12)
    cv2.putText(img, pl_text, (2300, 450), cv2.FONT_HERSHEY_PLAIN, 7, [255,255,255], 12)

    

def getTime(pathRawImg, dateFormat):
    
    if dateFormat == 'clocktime':
        return datetime.fromtimestamp(getctime(pathRawImg)).strftime('%H:%M:%S')
    elif dateFormat == 'seconds':
        h = datetime.fromtimestamp(getctime(pathRawImg)).strftime('%H')
        m = datetime.fromtimestamp(getctime(pathRawImg)).strftime('%M')
        s = datetime.fromtimestamp(getctime(pathRawImg)).strftime('%S')
        return int(h)*3600 + int(m)*60 + int(s)
    elif dateFormat == 'date':
        return datetime.fromtimestamp(getctime(pathRawImg)).strftime('%d.%m.%y')
    else:
        pass


checkDetection()
database = {}

for imgName in listdir(pathRawImg):
    rawImg = cv2.imread('%s\\%s'% (pathRawImg,imgName))
    plateImg = cv2.imread('%s\\det_%s'% (pathDetImg,imgName))
    time_clocktime = getTime('%s\\%s'% (pathRawImg,imgName), 'clocktime')
    time_seconds = getTime('%s\\%s'% (pathRawImg,imgName), 'seconds')
    time_date = getTime('%s\\%s'% (pathRawImg,imgName), 'date')
    
    
    grayImg = cv2.cvtColor(plateImg, cv2.COLOR_BGR2GRAY)
    plateToRead = Image.fromarray(grayImg) 
    gray_plateText = ocr(plateToRead)
        
    if len(gray_plateText) > 8  or len(gray_plateText) < 7:
        binImg = cv2.adaptiveThreshold(grayImg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,79,69)
        negBinImg = cv2.bitwise_not(binImg)
        matrix = np.ones((3,3), np.uint8)
        negBinImg = cv2.dilate(negBinImg,matrix,iterations = 1)
        negBinImg = cv2.morphologyEx(negBinImg, cv2.MORPH_OPEN, matrix)
        negBinImg = cv2.dilate(negBinImg,matrix,iterations = 1)
        negBinImg = cv2.erode(negBinImg,matrix,iterations = 1)
        
        plateToRead = Image.fromarray(negBinImg)
        
        negBin_plateText = ocr(plateToRead)
        
        if getLenOfLastWord(negBin_plateText) > 5:
            plateText = negBin_plateText[0:len(negBin_plateText)-1]
            if getLenOfLastWord(plateText) > 5:
                plateText = plateText[0:len(plateText)-1]
        
        elif len(negBin_plateText) <= 9 and len(negBin_plateText) >= 7:
            plateText = negBin_plateText
            
        else: 
            plateText = 'error'
            print('error: ', len(negBin_plateText))
    else:
        plateText = gray_plateText



    if plateText not in database:
        database[plateText] = time_seconds
    else:
        time_drive = time_seconds - database[plateText]
        del database[plateText]
        if time_drive < time_limit:
            cv2.namedWindow('Speed Result',cv2.WINDOW_NORMAL) 
            rawImg = drawRectInfobox(rawImg, 'red')
            putTextInfobox(rawImg, time_drive, plateText, str(imgName), time_date, time_clocktime)
            cv2.imshow('Speed Result',rawImg)
            cv2.waitKey()
            cv2.resizeWindow('Speed Result', 800,533)
            
        else:
            cv2.namedWindow('Speed Result',cv2.WINDOW_NORMAL) 
            rawImg = drawRectInfobox(rawImg, 'green')
            putTextInfobox(rawImg, time_drive, plateText, str(imgName), time_date, time_clocktime)
            cv2.imshow('Speed Result',rawImg)
            cv2.waitKey()
            cv2.resizeWindow('Speed Result', 800,533)
    
cv2.destroyAllWindows()   
