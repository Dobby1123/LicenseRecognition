import cv2 as cv

FPS = 25          # 每秒25帧
WINDOW_TITLE = "Camera Capture"
FRAME_WIDTH = 960
FRAME_HEIGHT = 720

cameraCapture = cv.VideoCapture()

def init():
    cameraCapture.open(0)              # 对于有多个摄像头计算机，0、1是摄像头的编号

    if not cameraCapture.isOpened():
        print("无法打开摄像头！")
        return False
    
    for i in range(5):               # 过滤掉前若干帧画面。这些画面因为设备初始化尚未完全完成等原因，拍摄出来的照片不正常
        cameraCapture.read()           
   
    # 在open之后才能获得摄像头拍照的宽、高和帧频率
    width = cameraCapture.get(cv.CAP_PROP_FRAME_WIDTH)        
    height = cameraCapture.get(cv.CAP_PROP_FRAME_HEIGHT)
    fps = cameraCapture.get(cv.CAP_PROP_FPS)
    # 如果当前驱动不支持获取某个属性值，则该属性值返回0
    print("摄像头捕获宽度：%d, 高度：%d, FPS:%d" % (width, height, fps))

    cameraCapture.set(cv.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cameraCapture.set(cv.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    return True

def process():
    success, frame = cameraCapture.read()
    cv.imshow(WINDOW_TITLE, frame)

init()
cv.namedWindow(WINDOW_TITLE)
interval = 1000 // FPS      # 计算每张照片采集的时间间隔(毫秒)
while True:
    key = cv.waitKey(interval)
    if key > 0:             # 如果按下任意键，则退出循环
        break
    process()
cv.destroyAllWindows()