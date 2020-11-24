import numpy as np
import cv2

def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[0]
    triangle = np.array([[(250, height), (1200, height), (650, 410)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255) 
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


video = cv2.VideoCapture("CH.mp4")
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('project1.avi', cv2.VideoWriter_fourcc('M', 'J',  'P', 'G'), 10, (width, height))
while True:
    ret, frame = video.read()
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    canny = cv2.Canny(blur, 75, 150)
    cropped_im = region_of_interest(canny)
    kernel = np.ones((5, 5), np.uint8)
    # kernel = np.zeros((5, 5), dtype=np.uint8)
    # kernel[:, 2] = 1
    # kernel[:, 2] = 1
    dilated_im = cv2.dilate(cropped_im, kernel)
    lines = cv2.HoughLinesP(dilated_im, 2, np.pi / 180, 200, minLineLength=80, maxLineGap=100)
    res = frame.copy()
    if lines is not None:
        for r_t in lines:
            x1, y1, x2, y2 = r_t[0]
            slope = (y2-y1)/(x2-x1)
            if slope > 0.5 or slope <-0.5:
               res = cv2.line(res, (x1, y1), (x2, y2), (0, 0, 255), 3)

    cv2.imshow("result", res)
    if cv2.waitKey(1) == ord('s'):
        break
video.release()
cv2.destroyAllWindows()
