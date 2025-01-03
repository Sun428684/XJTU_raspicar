import time
import numpy as np
import cv2

'''
标记颜色块中点程序,这里是黑线
'''

# 打开摄像头，0通常是默认摄像头的索引
cap = cv2.VideoCapture(0)
# 设置目标分辨率
target_resolution = (640, 480)


def find_line_midpoint(frame):
    # 转换到灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 应用阈值处理来创建掩码，这里假设黑线的灰度值较低
    _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)  # 阈值可以根据实际情况调整
    # 形态学操作，去除噪点
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)  # 使用闭运算来填充小的孔洞
    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 假设最长的轮廓是黑线
    max_area = 0
    best_cnt = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            best_cnt = cnt
            # 如果找到轮廓，则计算中点
    if best_cnt is not None:
        # 计算轮廓的边界矩形
        x, y, w, h = cv2.boundingRect(best_cnt)
        # 绘制矩形框（注意：这里的y是矩形框在原图上的顶部坐标）
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 使用蓝色绘制矩形框
        M = cv2.moments(best_cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cX, cY), 5, (0, 255, 0), -1)  # 标记中点
            print("({},{})".format(cX, cY))

            # 显示结果
    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', mask)


while True:
    ret, frame = cap.read()
    if not ret:
        break
    find_line_midpoint(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # 释放资源和关闭所有窗口
cap.release()
cv2.destroyAllWindows()