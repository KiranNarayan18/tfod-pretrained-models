from detector import *

# modelURL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d3_coco17_tpu-32.tar.gz'

modelURL= 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz'
threshold=0.5
classFiles = 'coco.names'
imagePath = 'cycle.jpg'
videoPath = 'pets.mp4'

detector = Detector()
detector.readClasses(classFiles)

detector.downloadModel(modelURL)

detector.loadModel()

# detector.predictImage(imagePath, threshold)

detector.predictVideos(videoPath, threshold)