
Start realtime 21 object detection
./darknet detector demo cfg/coco.data cfg/yolo.cfg yolo.weights


Start realtime fish detection 
./darknet detector demo -c 1 data/fish/fish.data data/fish/yolo-fish.2.0.cfg yolo-fish_final.weights

train fish dataset with darknet 
./darknet detector train data/fish/fish.data data/fish/yolo-fish-train.cfg darknet19_448.conv.23

train hand dataset with darknet
./darknet detector train data/hand/hand.data data/hand/yolo-hand-train.cfg darknet19_448.conv.23

#In training mode, the 'opencv' FLAG must set to 0 in MakeFile
#install sudo apt-get install libopencv-dev & python-opencv






















We train other dataset by using darknet

Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

