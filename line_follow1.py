import cv2
import numpy as np
import RPi.GPIO as gpio
import time

flag1=0
flag2=0
# 定义引脚
pin1 = 12  # 左正
pin2 = 16  # 左反
pin3 = 18  # 右正
pin4 = 22  # 右反
ENA = 38
ENB = 40
# 设置gpio口为BOARD编号规范
gpio.setmode(gpio.BOARD)

# 设置gpio口为输出
gpio.setup(pin1, gpio.OUT)
gpio.setup(pin2, gpio.OUT)
gpio.setup(pin3, gpio.OUT)
gpio.setup(pin4, gpio.OUT)
gpio.setup(ENA, gpio.OUT)
gpio.setup(ENB, gpio.OUT)

# 设置PWM波,频率为50Hz
pwm1 = gpio.PWM(ENA, 50)#该函数代表ENA输出的是  脉冲方波，频率为50Hz
pwm2 = gpio.PWM(ENB, 50)

pwm1.start(0) #初始PWM输出的信号，占空比为0，小车不动
pwm2.start(0)

def car_forward():  # 定义前进函数
    gpio.output(pin1, gpio.HIGH)  # 将pin1接口设置为高电压
    gpio.output(pin2, gpio.LOW)  # 将pin2接口设置为低电压
    gpio.output(pin3, gpio.HIGH)  # 将pin3接口设置为高电压
    gpio.output(pin4, gpio.LOW)  # 将pin4接口设置为低电压
def car_stop():  # 定义停止函数
    gpio.output(pin1, gpio.LOW)
    gpio.output(pin2, gpio.LOW)
    gpio.output(pin3, gpio.LOW)
    gpio.output(pin4, gpio.LOW)
def car_left():  # 定义左转函数
    gpio.output(pin1, gpio.LOW)
    gpio.output(pin2, gpio.HIGH)
    gpio.output(pin3, gpio.HIGH)
    gpio.output(pin4, gpio.LOW)
def car_right():  # 定义右转函数
    gpio.output(pin1, gpio.HIGH)
    gpio.output(pin2, gpio.LOW)
    gpio.output(pin3, gpio.LOW)
    gpio.output(pin4, gpio.HIGH)
def find_red(frame):
    red_result = 0
    #将图像转换为HSV空间
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #定义红色的HSV范围
    lower_red1 = np.array([0,100,100])
    upper_red1 = np.array([10,255,255])
    lower_red2 = np.array([170,100,100])
    upper_red2 = np.array([180,255,255])
    #创建红色掩膜
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = mask1 + mask2
    if np.sum(red_mask)>0:
        red_result = 1
    else:
        red_result = 0
    return red_result
def find_line_midpoint_in_roi(frame):
    # 随便给这三个用到的需要返回的值赋初始值，防止返回报错！
    detect = 3  # detect=0没检测到黑线，detect=1检测到黑线
    dx = 3
    dy = 3
    w=0
    RED=2
    # 获取帧的高度和宽度
    height, width = frame.shape[:2]
    # 计算底部四分之一区域的高度
    roi_height = height // 4
    # 定义ROI区域（底部四分之一）
    roi_y = height - roi_height
    roi = frame[roi_y:height, 0:width]
    RED=find_red(roi)
    # 转换ROI到灰度图像
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # 应用阈值处理来创建掩码
    _, mask = cv2.threshold(gray_roi, 50, 255, cv2.THRESH_BINARY_INV)
    # 形态学操作，去除噪点（可选）
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 查找轮廓
    _,contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 假设最长的轮廓是黑线
    max_area = 0
    best_cnt = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            best_cnt = cnt

            # 如果找到轮廓，则计算中点，并在原图上标记
    # 如果找到轮廓，则计算中点，并使用矩形框在原图上框出识别到的黑色块
    if best_cnt is not None:
        detect = 1
        # 注意：轮廓点坐标是相对于mask的，但我们需要将其转换为原图的坐标
        # 假设best_cnt是一个轮廓点集

        # 计算轮廓的边界矩形（这一步是在ROI坐标系中进行的）
        x, y, w, h = cv2.boundingRect(best_cnt)

        # 将边界矩形的坐标从ROI坐标系转换为原图坐标系
        # 注意：这里x坐标（即边界矩形的左边界）不需要修改，因为它已经是相对于原图的左边界
        # 而y坐标（即边界矩形的下边界，因为ROI是从底部开始的）需要转换为原图的坐标
        y_original = height - (y + h)  # 将ROI中的下边界转换为原图的上边界

        # 现在我们可以在原图上绘制矩形了
        cv2.rectangle(frame, (x, y_original), (x + w, y_original + h), (0, 0, 255), 2)  # 使用红色绘制矩形框

        # 接下来计算并绘制轮廓的中心点
        M = cv2.moments(best_cnt)
        cX = int(M["m10"] / M["m00"])  # 轮廓中心的X坐标（相对于ROI左上角），无需修改
        cY_roi = int(M["m01"] / M["m00"])  # 轮廓中心的Y坐标（相对于ROI底部）
        cY = height - cY_roi  # 将Y坐标从ROI底部转换为原图顶部
        cv2.circle(frame, (cX, cY), 5, (0, 255, 0), -1)  # 在原图上绘制圆心

        # 打印中心点坐标（可选）
        print("({},{})".format(cX, cY))

        # 计算 ROI 的中心点
        roi_mid_x = width // 2  # ROI 的 X 中心点（因为 ROI 总是从原图底部开始的全宽）
        roi_mid_y = roi_y + roi_height // 2  # ROI 的 Y 中心点（注意要加上 ROI 的起始 Y 坐标）
        # 绘制 ROI 中心点
        cv2.circle(frame, (roi_mid_x, roi_mid_y), 5, (255, 0, 0), -1)  # 使用蓝色绘制 ROI 中心点
        # 计算并绘制连接线
        cv2.line(frame, (roi_mid_x, roi_mid_y), (cX, cY), (0, 0, 255), 2)  # 使用红色绘制连接线
        # 计算并打印两个点的 X 和 Y 差值
        dx = cX - roi_mid_x
        dy = cY - roi_mid_y
        # 创建包含差值的文本字符串
        text = f"dx={dx}, dy={dy}"
        # text_x 与 text_y 表示文本打印的位置
        text_x = int((roi_mid_x + cX) / 2)  # 线的中点 X 坐标（可能需要根据实际情况调整）
        text_y = int((roi_mid_y + cY) / 2 - 10)  # 线的中点 Y 坐标减去一些空间以容纳文本
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    elif best_cnt is None:  # 没找到黑线
        detect = 0

        # 返回处理后的帧（可选，也可以直接在原图上操作）
    return frame, mask, dx, dy, detect,w,RED


# 主循环，用于从摄像头读取帧并处理
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 使用下划_忽略接收最后俩个返回的dx，dy的差值
    frame, mask, dx, _, detect ,w,red= find_line_midpoint_in_roi(frame)

    if(detect==0):
        car_stop()
    if(flag1==0):
        if(w>300):
            car_stop()
            flag1=1
            time.sleep(1)
    if (flag2 == 0):
        if (red==1):
            car_stop()
            flag2 = 1
            time.sleep(3)
    if(detect==1):
        if(dx<-50):
            car_forward()
            pwm1.ChangeDutyCycle(0)
            pwm2.ChangeDutyCycle(38)
        elif(dx>50):
            car_forward()
            pwm1.ChangeDutyCycle(38)
            pwm2.ChangeDutyCycle(0)
        else:
            car_forward()
            pwm1.ChangeDutyCycle(40)
            pwm2.ChangeDutyCycle(40)
    time.sleep(0.15)
    # 显示结果
    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', mask)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


pwm1.ChangeDutyCycle(0)
pwm2.ChangeDutyCycle(0)
cap.release()
cv2.destroyAllWindows()