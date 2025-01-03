import cv2
import numpy as np


def find_line_midpoint_in_roi(frame):
    # 随便给这三个用到的需要返回的值赋初始值，防止返回报错！
    detect = 3  # detect=0没检测到黑线，detect=1检测到黑线
    dx = 3
    dy = 3
    # 获取帧的高度和宽度
    height, width = frame.shape[:2]
    # 计算底部四分之一区域的高度
    roi_height = height // 4
    # 定义ROI区域（底部四分之一）
    roi_y = height - roi_height
    roi = frame[roi_y:height, 0:width]
    # 转换ROI到灰度图像
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # 应用阈值处理来创建掩码
    _, mask = cv2.threshold(gray_roi, 50, 255, cv2.THRESH_BINARY_INV)
    # 形态学操作，去除噪点（可选）
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

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
        #print("({},{})".format(cX, cY))

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
        print("({},{})".format(dx, dy))
        # 创建包含差值的文本字符串
        text = f"dx={dx}, dy={dy}"
        # text_x 与 text_y 表示文本打印的位置
        text_x = int((roi_mid_x + cX) / 2)  # 线的中点 X 坐标（可能需要根据实际情况调整）
        text_y = int((roi_mid_y + cY) / 2 - 10)  # 线的中点 Y 坐标减去一些空间以容纳文本
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    elif best_cnt is None:  # 没找到黑线
        detect = 0

        # 返回处理后的帧（可选，也可以直接在原图上操作）
    return frame, mask, dx, dy, detect


# 主循环，用于从摄像头读取帧并处理
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 使用下划_忽略接收最后俩个返回的dx，dy的差值
    frame, mask, _, _, _ = find_line_midpoint_in_roi(frame)

    # 显示结果
    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', mask)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()