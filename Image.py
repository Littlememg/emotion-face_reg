from rmn import RMN
import cv2
m = RMN()

image = cv2.imread("TEST1.jpg")
results = m.detect_emotion_for_single_frame(image)
print('Name: ', results[0]['name'])
print('Emotion: ', results[0]['emo_label'])
image = m.draw(image, results)
cv2.imwrite("output.png", image)