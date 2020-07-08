
@CALL "%~dp0..\condabin\conda.bat" activate %*
cd C:\Users\park3r\Anaconda3

activate yolov3 && cd C:\Users\park3r\Anaconda3\yolo\pytorch-yolo-v3 && python detect.py --images img_auto --det detz_auto --reso 256 && deactivate && exit

