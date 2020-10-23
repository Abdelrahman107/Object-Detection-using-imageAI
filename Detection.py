from imageai.Detection import ObjectDetection
import os
 

path1 = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet() #download resnet50_coco_best_v2.0.1.h5 pretrained model 
detector.setModelPath(os.path.join(path1, "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(path1, "image2.jpg"), output_image_path=os.path.join(path1, "image2Detected.jpg"))

for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )