import cv2

imagePath = 'image3.jpg'
cascadeClassifierPath ='haarcascade_frontalface_alt.xml'

casecadeClassifier = cv2.CascadeClassifier(cascadeClassifierPath)
image = cv2.imread(imagePath)
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

detectedFaces = casecadeClassifier.detectMultiScale(grayImage)

for(x,y, width, height) in detectedFaces:
    cv2.rectangle(image, (x,y), (x*width , y*height ), ( 0, 0, 255), 10)
    
cv2. imwrite('output.jpg', image)