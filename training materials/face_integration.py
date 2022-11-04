import cv2,time,os
import mediapipe as mp
import numpy as np
from requests import patch
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
def intdef(user_input):
    try:
        int(user_input)
        intnum = True
    except ValueError:
        intnum = False
    return intnum
def timer(settime):
    nowtime=time.time()
    while True:
        if time.time() - nowtime > settime:
            break

def data(data_Folder:int,num):
    time1 = time.time()
    i=1
    a=0
    while True:
        filepath = f"./Data{a}"
        if os.path.isdir(filepath):
            print(f"已找到資料{filepath}")
            a=a+1
        else:
            a=a-1
            break

    if data_Folder == 0:
        b=1
        filepath = f"./Data{a}"
        filenum = len(os.listdir(filepath))
        print("覆蓋資料中...")
        for files in range(1,filenum+1):
            if os.path.isfile(f"{filepath}/{files}.jpg"):
                os.remove(f"{filepath}/{files}.jpg")
                print('\r' + f'[正在覆蓋]:[%s%s],{files}/{filenum}' % ('█' * int(files) , ' ' * int(filenum-files)), end='')
                b=b+1
            else:
                print()
                print("資料已清除完成")
                break
        print()
    else:
        a = a+1
        filepath = f"./Data{a}"
        print("建立新資料中...")
        os.mkdir(filepath)
        print("完成")
    print("五秒後啟動相機進行捕捉")
    for n in range(6):
        print('\r'+f"{5-n}", end="")
        timer(1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("錯誤:相機未啟動")
        exit()

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0) as face_detection:
            while True:
                success, image = cap.read()
                img=image
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_detection.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.detections:  #是否出現人臉
                    for detection in results.detections:
                        mp_drawing.draw_detection(img, detection)
                        bounding_box = detection.location_data.relative_bounding_box
                        x = int(bounding_box.xmin * image.shape[1])
                        w = int(bounding_box.width * image.shape[1])
                        y = int(bounding_box.ymin * image.shape[0])
                        h = int(bounding_box.height * image.shape[0])
                        cv2.rectangle(image, (x, y), (x + w, y + h), (210, 150, 150), thickness = 2)
                        cv2.putText(img,f"{i}/{num}",(350,30),cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,225,0),2)
                        if time.time() - time1 > 1: #判斷上一次紀錄的時間戳減去當前時間戳是否超過5秒
                            time1 = time.time()
                            cv2.imwrite(f'./{filepath}/{i}.jpg', image) #暫存圖片
                            print('\r' + f'[目前進度]:[%s%s],{i}/{filenum}' % ('█' * int(i) , ' ' * int(num-i)), end='')
                            i=i+1
                Data = False
                if i==num+1:
                    cv2.putText(img,f"Data OK",(100,30),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,225,0),2)
                    print()
                    print("資料獲取完成")
                    Data=True
                cv2.imshow('Face Detection',img)
                if Data == True:
                    break


                if cv2.waitKey(1) == 32:
                  break
    print("整理中...")

def taining():
    detector = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')  # 載入人臉追蹤模型
    recog = cv2.face.LBPHFaceRecognizer_create()      # 啟用訓練人臉模型方法
    faces = []   # 儲存人臉位置大小的串列
    ids = []     # 記錄該人臉 id 的串列
    print('訓練中...')                              # 提示開始訓練
    
    pathdata = 0
    while True:
        filepath = f"./Data{pathdata}"
        if os.path.isdir(filepath):
            filenum = len(os.listdir(filepath))
            print(f"{filepath}路徑總資料量:{filenum}樣")
            for i in range(1,filenum+1):
                img = cv2.imread(f'{filepath}/{i}.jpg')       # 依序開啟每一張再資料夾裡的照片
                print('\r' + f'[正在讀取]:[%s%s],{i}/{filenum}' % ('█' * int(i) , ' ' * int(filenum-i)), end='')
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 色彩轉換成黑白
                img_np = np.array(gray,'uint8')               # 轉換成指定編碼的 numpy 陣列
                face = detector.detectMultiScale(gray)        # 擷取人臉區域
                for(x,y,w,h) in face:
                    faces.append(img_np[y:y+h,x:x+w])         # 記錄人臉的位置和大小內像素的數值
                    ids.append(pathdata+1)                      # 記錄人臉對應的 id，只能是整數，都是 1 表示該人臉的 id 為 1
            print()
            pathdata = pathdata+1
        else:
            print('確認寫入完成')
            break


    recog.train(faces,np.array(ids))                  # 開始訓練
    recog.save('./face.yml')       # 訓練完成儲存為 face.yml
    print('完成')



print("偵測整合")

while True:

    data_collection = str(input("是否需要進行資料收集(y/n):"))
    if data_collection == "y" or data_collection == "Y":
        data_Folder = int(input("覆蓋上一次資料或者新增資料(覆蓋=0/新增=1):"))
        while True:
            if data_Folder == 0:
                print("已確認選擇覆蓋...")
                data_quantity = (input("請選擇新增多少筆資料(最多100):"))
                intnum=intdef(data_quantity)
                if intnum == True:
                    data_quantity=int(data_quantity)
                    if 0<data_quantity<101:
                        print("啟動中...")
                        data(data_Folder,data_quantity)
                    else:
                        print("錯誤:參數範圍錯誤(1~100)")
                else:
                    print("錯誤:非int數")
                
            elif data_Folder == 1:
                print("已確認選擇新增...")
                data_quantity = (input("請選擇新增多少筆資料(最多100):"))
                intnum=intdef(data_quantity)
                if intnum == True:
                    data_quantity=int(data_quantity)
                    if 0<data_quantity<101:
                        print("啟動中...")
                        data(data_Folder,data_quantity)
                    else:
                        print("錯誤:燦樹範圍錯誤(1~100)")
                else:
                    print("錯誤:非int數")
             
            else:
                print("範圍外,請重新輸入")
                   
            time.sleep(3)
            tainingconfirm = str(input("是否馬上進行資料訓練(y/n):"))
            if tainingconfirm == "y" or tainingconfirm == "Y":
                print("啟動中...")
                taining()
                break
            if tainingconfirm == "n" or tainingconfirm == "N":
                print("已關閉...")
                break
            continue
    elif data_collection=="n"or data_collection=="N":
        tainingconfirm = str(input("是否使用舊有的資料進行訓練(y/n):"))
        if tainingconfirm == "y" or tainingconfirm == "Y":
            print("啟動中...")
            taining()
            break
        if tainingconfirm == "n" or tainingconfirm == "N":
            print("已關閉...")
            break
    else:
        print("範圍外,請重新輸入")
        continue
