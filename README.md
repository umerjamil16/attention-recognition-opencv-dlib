# Attention Recognition using OpenCV
Please see the pdf report where I have explained the key steps and the theory of implementing an attention recognition systemunder different light conditions. The implemented program is able to
- Tack eye ball (up, down, left, right, center focus)
- Blinking (eyes open/ close).3. Labels facial parts (such as Mouth, Nose and Jaw)
- Perform under different lighting condition
 
## Environment
I have used Python in PyCharm environment. Other libraries that I used are OpenCV and Dlib. The respective versions are as follows:
- Python 3.6
- OpenCV 3.4.2
- Dlib 19.8.2
# Background
## Face Recognition
Face recognition means “given an arbitrary image, the goal of face detection is to determine whether or notthere are any faces in the image and, if present, return the image location and extent of each”. Practically,face detection algorithms will aim at generating bounding boxes (often rectangular or elliptical) around all thefaces in the image and only around facesSome of the popular face recognition techniques are as follows:
- Viola & Jones Haar-cascade algorithm
- Speeded-Up Robust Features (SURF)
- Local binary patterns (LBP)
- Histogram of Oriented Gradients (HOG)
- Convolution Neural Network Based Models
## Attention Recognition
Tracking eyes has been a subject of research and development in many fields. In literature, attention recognitionis the process of measuring either the point of gaze (where one is looking) or the motion of an eye relative tothe head. Eye tracking is the measurement of eye movement/activity and gaze (point of regard) tracking isthe analysis of eye tracking data with respect to the head/visual scene. Researches of this field often use theterms eye-tracking, gaze-tracking or eye-gaze tracking interchangeably.We can categories the movement of eye ball into following major movements:
- Movement to the Right
- Movement to the Left
- Movement to the Up
- Movement to the Down
- Eyeball at center

License
----

MIT


