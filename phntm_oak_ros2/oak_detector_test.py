###
# ROSified version of https://docs.luxonis.com/software/depthai/examples/spatial_tiny_yolo/
# Using yolo-v4
# depthai-python installed in /root/depthai-python
###

from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import time
import argparse
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose
import math
import rclpy
from rclpy.node import Node, Parameter, QoSProfile, Publisher
from rclpy.qos import QoSHistoryPolicy, QoSReliabilityPolicy, DurabilityPolicy
from pprint import pprint
import json
import array
import ctypes

from sensor_msgs.msg import Image

from .inc.lib import set_message_header, msg_data_from_frame

def main():
    print('Hi from oak_detector_test')
    
    rclpy.init()
    rcl_node = Node(node_name='oak')
    
    det_3d_qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
    det_3d_pub = rcl_node.create_publisher(Detection3DArray, '/oak/nn/spatial_detections', det_3d_qos)
    
    depth_qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
    depth_pub = rcl_node.create_publisher(Image, '/oak/stereo/image_raw', depth_qos)
    
    rgb_prev_qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
    rgb_prev_pub = rcl_node.create_publisher(Image, '/oak/rgb/preview/image_raw', rgb_prev_qos)
    
    # labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",  "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    # nnPathDefault = '/ws/depthai-python/examples/models/mobilenet-ssd_openvino_2021.4_5shave.blob'
    labelMap = [ 'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'potted plant', 'bed', 'dining table', 'toilet', 'tv monitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush' ] 
    nnBlobPath = '/ros2_ws/src/phntm_oak_ros2/models/tiny-yolo-v4_openvino_2021.2_6shave.blob'
    
    syncNN = False

    # Create pipeline
    pipeline = dai.Pipeline()

    # Define sources and outputs
    camRgb = pipeline.create(dai.node.ColorCamera)
    spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    nnNetworkOut = pipeline.create(dai.node.XLinkOut)

    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutNN = pipeline.create(dai.node.XLinkOut)
    xoutDepth = pipeline.create(dai.node.XLinkOut)

    xoutRgb.setStreamName("rgb")
    xoutNN.setStreamName("detections")
    xoutDepth.setStreamName("depth")
    nnNetworkOut.setStreamName("nnNetwork")

    # Properties
    camRgb.setPreviewSize(416, 416)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    monoLeft.setCamera("left")
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    monoRight.setCamera("right")

    # setting node configs
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
    # Align depth map to the perspective of RGB camera, on which inference is done
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())
    stereo.setSubpixel(False)
    stereo.initialConfig.setConfidenceThreshold(200)
    stereo.initialConfig.setLeftRightCheckThreshold(5)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(True)
    
    spatialDetectionNetwork.setBlobPath(nnBlobPath)
    spatialDetectionNetwork.setConfidenceThreshold(0.5)
    spatialDetectionNetwork.input.setBlocking(False)
    spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
    spatialDetectionNetwork.setDepthLowerThreshold(100)
    spatialDetectionNetwork.setDepthUpperThreshold(5000)

    # Yolo specific parameters
    spatialDetectionNetwork.setNumClasses(80)
    spatialDetectionNetwork.setCoordinateSize(4)
    spatialDetectionNetwork.setAnchors([10,14, 23,27, 37,58, 81,82, 135,169, 344,319])
    spatialDetectionNetwork.setAnchorMasks({ "side26": [1,2,3], "side13": [3,4,5] })
    spatialDetectionNetwork.setIouThreshold(0.5)
    spatialDetectionNetwork.setNumInferenceThreads(1)

    # Linking
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    camRgb.preview.link(spatialDetectionNetwork.input)
    # if syncNN:
    #     spatialDetectionNetwork.passthrough.link(xoutRgb.input)
    # else:
    #     camRgb.preview.link(xoutRgb.input)
    camRgb.preview.link(xoutRgb.input)

    spatialDetectionNetwork.out.link(xoutNN.input)

    stereo.depth.link(spatialDetectionNetwork.inputDepth)
    spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)
    spatialDetectionNetwork.outNetwork.link(nnNetworkOut.input)

    try:
        
        # Connect to device and start pipeline
        with dai.Device(pipeline) as device:

            # Print MxID, USB speed, and available cameras on the device
            print('MxId:', device.getDeviceInfo().getMxId())
            print('USB speed:', device.getUsbSpeed())
            print('Connected cameras:', device.getConnectedCameras())
            
            # Output queues will be used to get the rgb frames and nn data from the outputs defined above
            previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
            depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
            networkQueue = device.getOutputQueue(name="nnNetwork", maxSize=4, blocking=False)

            startTime = time.monotonic()
            counter = 0
            fps = 0
            color = (255, 255, 255)
            printOutputLayersOnce = True
        
            while True:
                
                det_3d_msg = Detection3DArray()
                set_message_header('oak_rgb_camera_optical_frame', det_3d_msg)
                det_3d_msg.detections = []
                
                rgb_prev_msg = Image()
                set_message_header('', rgb_prev_msg)
                
                depth_msg = Image()
                set_message_header('', depth_msg)
                
                inPreview = previewQueue.get()
                inDet = detectionNNQueue.get()
                depth = depthQueue.get()
                inNN = networkQueue.get()

                if printOutputLayersOnce:
                    toPrint = 'Output layer names:'
                    for ten in inNN.getAllLayerNames():
                        toPrint = f'{toPrint} {ten},'
                    print(toPrint)
                    printOutputLayersOnce = False

                frame = inPreview.getCvFrame()
                depthFrame = depth.getFrame() # depthFrame values are in millimeters

                msg_data_from_frame(inPreview, frame, rgb_prev_msg)
                msg_data_from_frame(depth, depthFrame, depth_msg)
                # pprint(inPreview.getData().tolist())
                #rgb_prev_msg.data = frame.flatten().tolist()
                #if rgb_buf == None:
                #    rgb_buf = array.array('B', [0] * (rgb_prev_msg.width * rgb_prev_msg.height * 3))
                #ctypes.memmove(rgb_buf, frame.ravel(), rgb_prev_msg.width * rgb_prev_msg.height * 3)
                # np.copyto(rgb_buf, frame.flatten())
                #ctypes.memmove(rgb_buf, frame.flatten(), rgb_prev_msg.width * rgb_prev_msg.height * 3)
                #frame.flatten() #.tobytes('C', rgb_buf.buffer_info()[0])
                # rgb_prev_msg = frame.tolist()

                depth_downscaled = depthFrame[::4]
                if np.all(depth_downscaled == 0):
                    min_depth = 0  # Set a default minimum depth value when all elements are zero
                else:
                    min_depth = np.percentile(depth_downscaled[depth_downscaled != 0], 1)
                max_depth = np.percentile(depth_downscaled, 99)
                #depthFrameColor = np.interp(depthFrame, (min_depth, max_depth), (0, 255)).astype(np.uint8)
                #depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

                counter+=1
                current_time = time.monotonic()
                if (current_time - startTime) > 1 :
                    fps = counter / (current_time - startTime)
                    counter = 0
                    startTime = current_time

                detections = inDet.detections

                # If the frame is available, draw bounding boxes on it and show the frame
                height = frame.shape[0]
                width  = frame.shape[1]
                for detection in detections:
                    roiData = detection.boundingBoxMapping
                    roi = roiData.roi
                    roi = roi.denormalize(depthFrame.shape[1], depthFrame.shape[0])
                    topLeft = roi.topLeft()
                    bottomRight = roi.bottomRight()
                    xmin = int(topLeft.x)
                    ymin = int(topLeft.y)
                    xmax = int(bottomRight.x)
                    ymax = int(bottomRight.y)
                    #cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, 1)

                    # Denormalize bounding box
                    x1 = int(detection.xmin * width)
                    x2 = int(detection.xmax * width)
                    y1 = int(detection.ymin * height)
                    y2 = int(detection.ymax * height)
                    try:
                        label = labelMap[detection.label]
                    except:
                        label = detection.label
            
                    xSize = x2 - x1
                    ySize = y2 - y1
                    xCenter = x1 + xSize / 2.0
                    yCenter = y1 + ySize / 2.0
                    
                    # json.dumps(det)
                    #print(f'{label} {detection.confidence:.2f} [{x1}, {y1}, {x2}, {y2}] => [{int(detection.spatialCoordinates.x)}; {int(detection.spatialCoordinates.y)}; {int(detection.spatialCoordinates.z)}]')
                      
                    # generate ros msg    
                    res = Detection3D()
                    
                    res.bbox.center.position.x = xCenter
                    res.bbox.center.position.y = yCenter
                    res.bbox.size.x = float(xSize)
                    res.bbox.size.y = float(ySize)
                    res.bbox.size.z = 0.01
        
                    res.results = []
                    
                    hyp = ObjectHypothesisWithPose()
                    hyp.hypothesis.class_id = str(detection.label)
                    hyp.hypothesis.score = detection.confidence
                    
                    hyp.pose.pose.position.x = detection.spatialCoordinates.x / 1000.0 # mm => m
                    hyp.pose.pose.position.y = -1.0 * detection.spatialCoordinates.y / 1000.0 # UPSIDE DOWN!
                    hyp.pose.pose.position.z = detection.spatialCoordinates.z / 1000.0
                    
                    res.results.append(hyp)
                    det_3d_msg.detections.append(res)
                        
                #print('---')
                if rcl_node.context.ok():
                    det_3d_pub.publish(det_3d_msg)
                    rgb_prev_pub.publish(rgb_prev_msg)
                    depth_pub.publish(depth_msg)
                
    except KeyboardInterrupt:
        print('Stopping...')
    
    # rclpy.shutdown()
    # rcl_node.destroy_node()

if __name__ == '__main__':
    main()
