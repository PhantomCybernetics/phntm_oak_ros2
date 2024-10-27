from pathlib import Path
#import cv2
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

def main():
    print('Hi from oak_tracker_test')
    
    rclpy.init()
    rcl_node = Node(node_name='oak_test')
    qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST,
                                depth=1,
                                reliability=QoSReliabilityPolicy.BEST_EFFORT
                                )
    rcl_pub = rcl_node.create_publisher(Detection3DArray, '/oak/nn/spatial_detections', qos)
    
    # labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",  "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    # nnPathDefault = '/ws/depthai-python/examples/models/mobilenet-ssd_openvino_2021.4_5shave.blob'
    labelMap = [ 'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'potted plant', 'bed', 'dining table', 'toilet', 'tv monitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush' ] 
    nnPathDefault = '/root/depthai-python/examples/models/yolo-v4-tiny-tf_openvino_2021.4_6shave.blob'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('nnPath', nargs='?', help="Path to YOLO detection network blob", default=nnPathDefault)
    parser.add_argument('-ff', '--full_frame', action="store_true", help="Perform tracking on full RGB frame", default=False)

    args = parser.parse_args()

    fullFrameTracking = False

    # Create pipeline
    pipeline = dai.Pipeline()

    # Define sources and outputs
    camRgb = pipeline.create(dai.node.ColorCamera)
    spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    
    stereo = pipeline.create(dai.node.StereoDepth)
    objectTracker = pipeline.create(dai.node.ObjectTracker)

    xoutRgb = pipeline.create(dai.node.XLinkOut)
    trackerOut = pipeline.create(dai.node.XLinkOut)

    xoutRgb.setStreamName("preview")
    trackerOut.setStreamName("tracklets")

    # Properties
    camRgb.setPreviewSize(416, 416)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setCamera("left")
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setCamera("right")

    # setting node configs
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    # Align depth map to the perspective of RGB camera, on which inference is done
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())

    spatialDetectionNetwork.setBlobPath(args.nnPath)
    spatialDetectionNetwork.setConfidenceThreshold(0.3)
    spatialDetectionNetwork.input.setBlocking(False)
    spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
    spatialDetectionNetwork.setDepthLowerThreshold(100)
    spatialDetectionNetwork.setDepthUpperThreshold(5000)
    spatialDetectionNetwork.setNumClasses(80)
    spatialDetectionNetwork.setCoordinateSize(4)
    spatialDetectionNetwork.setAnchors([10,14, 23,27, 37,58, 81,82, 135,169, 344,319])
    spatialDetectionNetwork.setAnchorMasks({ "side13": [3,4,5], "side26": [1,2,3] })
    spatialDetectionNetwork.setIouThreshold(0.5)

    # objectTracker.setDetectionLabelsToTrack([15])  # track only person
    # possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS, SHORT_TERM_IMAGELESS, SHORT_TERM_KCF
    objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_IMAGELESS)
    # take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
    objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.UNIQUE_ID)

    # Linking
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    camRgb.preview.link(spatialDetectionNetwork.input)
    objectTracker.passthroughTrackerFrame.link(xoutRgb.input)
    objectTracker.out.link(trackerOut.input)

    if fullFrameTracking:
        camRgb.setPreviewKeepAspectRatio(False)
        camRgb.video.link(objectTracker.inputTrackerFrame)
        objectTracker.inputTrackerFrame.setBlocking(False)
        # do not block the pipeline if it's too slow on full frame
        objectTracker.inputTrackerFrame.setQueueSize(2)
    else:
        spatialDetectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)

    spatialDetectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
    spatialDetectionNetwork.out.link(objectTracker.inputDetections)
    stereo.depth.link(spatialDetectionNetwork.inputDepth)

    # Connect to device and start pipeline
    try:
        with dai.Device(pipeline) as device:

            preview = device.getOutputQueue("preview", 4, False)
            tracklets = device.getOutputQueue("tracklets", 4, False)

            startTime = time.monotonic()
            counter = 0
            fps = 0
            color = (255, 255, 255)

            print (f'Connected to device')
             
            while(True):
                
                imgFrame = preview.get()
                track = tracklets.get()

                counter+=1
                current_time = time.monotonic()
                if (current_time - startTime) > 1 :
                    fps = counter / (current_time - startTime)
                    counter = 0
                    startTime = current_time

                frame = imgFrame.getCvFrame()
                trackletsData = track.tracklets
                
                msg = Detection3DArray()
                time_nanosec:int = time.time_ns()
                msg.header.stamp.sec = math.floor(time_nanosec / 1000000000)
                msg.header.stamp.nanosec = time_nanosec % 1000000000
                msg.header.frame_id = 'oak_rgb_camera_optical_frame'
                msg.detections = []
                    
                for t in trackletsData:
                    roi = t.roi.denormalize(frame.shape[1], frame.shape[0])
                    x1 = int(roi.topLeft().x)
                    y1 = int(roi.topLeft().y)
                    x2 = int(roi.bottomRight().x)
                    y2 = int(roi.bottomRight().y)
            
                    try:
                        label = labelMap[t.label]
                    except:
                        label = t.label
                    
                    img_det = t.srcImgDetection
                    
                    print(f'{t.id}: {label} {t.status.name}({t.status}) {img_det.confidence:.2f} [{x1}, {y1}, {x2}, {y2}] => [{int(t.spatialCoordinates.x)}; {int(t.spatialCoordinates.y)}; {int(t.spatialCoordinates.z)}]')
                    
                    if t.status != dai.Tracklet.TrackingStatus.TRACKED:
                        continue
                    
                    xSize = x2 - x1
                    ySize = y2 - y1
                    xCenter = x1 + xSize / 2.0
                    yCenter = y1 + ySize / 2.0
                    
                    # json.dumps(det)
                    
                    # generate ros msg    
                    res = Detection3D()
                    
                    res.bbox.center.position.x = xCenter
                    res.bbox.center.position.y = yCenter
                    res.bbox.size.x = float(xSize)
                    res.bbox.size.y = float(ySize)
                    res.bbox.size.z = 0.01
        
                    res.results = []
                    
                    hyp = ObjectHypothesisWithPose()
                    hyp.hypothesis.class_id = str(t.label)
                    hyp.hypothesis.score = img_det.confidence
                    
                    hyp.pose.pose.position.x = t.spatialCoordinates.x / 1000.0 # mm => m
                    hyp.pose.pose.position.y = -1.0 * t.spatialCoordinates.y / 1000.0 # UPSIDE DOWN!
                    hyp.pose.pose.position.z = t.spatialCoordinates.z / 1000.0
                    
                    res.results.append(hyp)
                    msg.detections.append(res)
                    
                print('---')
                if rcl_node.context.ok():
                    rcl_pub.publish(msg)
                

    except KeyboardInterrupt:
        print('Stopping...')
    
    # rclpy.shutdown()
    # rcl_node.destroy_node()

if __name__ == '__main__':
    main()
