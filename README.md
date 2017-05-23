
Start realtime 21 object detection
./darknet detector demo cfg/coco.data cfg/yolo.cfg yolo.weights


Start realtime fish detection 
./darknet detector demo -c 1 cfg/fish.data cfg/yolo-fish.2.0.cfg yolo-fish_final.weights

train with darknet 
./darknet detector train cfg/voc.data cfg/yolo-voc.cfg darknet19_448.conv.23


























We train other dataset by using darknet

Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

