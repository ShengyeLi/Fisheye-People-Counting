# Fisheye-People-Counting
This project contains the Activity-Blind(AB) and Activity-Aware(AA) applications of YOLOv3 on people counting using an overhead fisheye camera. The detail of these two methods is described in paper Supervised People Counting Using An Overhead Fisheye Camera, which are accepted in IEEE AVSS 2019.
The dataset used for experiment is organized by the Visual Information Processing group of Boston University.

AA.ipynb is the source code in form of python notebook for activity-aware application of YOLO v3 on people counting using fisheye camera, and AA.py is the same code in form of python code. 
AB.ipynb is the source code for activity-blind application, and AB.py is its corresponding python code. 

To run the code, you must import util.py and darkness.py. In AA.ipynb and Ab.ipynb, it is coded in an interactive way to import these two files: you need two run the first cell, select and upload util.py and darkness.py to colab. 

The util.py and darknet.py are obtained in the GitHub repo: https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch, and you can also find a really nice tutorial on implementing YOLO v3 in Pytorch via: https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/ This is also the framework on which my codes are developed based. If there is anything unclear in my comment about implementation of YOLO, please check the tutorial for reference.

Codes are developed and tested with:
	OpenCV 3.4.3
	PyTorch 1.0.1,post2
	Numpy 1.16.2
	Sklearn 0.20.3

The code is developed to utilize GPU for acceleration, so it has to run with a GPU version of PyTorch, if no revision is made.

To run the code, you need to 
1. Extract frames of the video you would like to test at first, because the code are only developed only for image processing. For colab, upload frames to google drive.
2. Import util.py and darknet,py.
3. Mount google drive for read and write file.
4. Revise the variable "readpath" to where extracted frames are, and set variable "savepath" to where you'd like to save result. 
5. Change the number of frames
6. run code.
