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
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose, Detection2DArray, Detection2D
import math
import rclpy
from rclpy.node import Node, Parameter, QoSProfile, Publisher
from rclpy.qos import QoSHistoryPolicy, QoSReliabilityPolicy, DurabilityPolicy
from pprint import pprint
import json
import array
import ctypes
import asyncio
import threading

from termcolor import colored as c

from sensor_msgs.msg import Image
from ffmpeg_image_transport_msgs.msg import FFMPEGPacket
from sensor_msgs.msg import CameraInfo

from .inc.lib import set_message_header, msg_data_from_frame, image_frame_loop, video_frame_loop, publisher_subscribed, msg_camera_info, adjust_intrinsics

async def async_loop():
    
    rcl_node = Node(node_name='oak')
    
    det_qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
    det_3d_pub = rcl_node.create_publisher(Detection3DArray, '/oak/nn/yolo8/spatial_detections', det_qos)
    det_2d_pub = rcl_node.create_publisher(Detection2DArray, '/oak/nn/yolo8/detections', det_qos)
    
    depth_qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
    depth_pub = rcl_node.create_publisher(Image, '/oak/stereo/image_raw', depth_qos)
    
    rgb_prev_qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
    rgb_prev_pub = rcl_node.create_publisher(Image, '/oak/rgb/preview/image_raw', rgb_prev_qos)
    
    rgb_h264_qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
    rgb_h264_pub = rcl_node.create_publisher(FFMPEGPacket, '/oak/rgb/image_raw/compressed', rgb_h264_qos)
    
    camera_info_qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1, reliability=QoSReliabilityPolicy.RELIABLE)
    rgb_info_pub = rcl_node.create_publisher(CameraInfo, '/oak/rgb/camera_info', camera_info_qos)
    prev_info_pub = rcl_node.create_publisher(CameraInfo, '/oak/rgb/preview/camera_info', camera_info_qos)
    left_info_pub = rcl_node.create_publisher(CameraInfo, '/oak/left/camera_info', camera_info_qos)
    right_info_pub = rcl_node.create_publisher(CameraInfo, '/oak/right/camera_info', camera_info_qos)

    nn_blob_path = '/ros2_ws/src/phntm_oak_ros2/models/tiny-yolo-v4_openvino_2021.2_6shave.blob'
    nn_inference_threads = 1 # keeps crashing with 2+
    nn_w = 416
    nn_h = 416
    
    # nn_blob_path = '/ros2_ws/src/phntm_oak_ros2/models/yolov8n_coco_640x352_5_shaves.blob'
    # nn_inference_threads = 2 # more threads for the encoder??
    # nn_w = 640
    # nn_h = 352
    
    # Create pipeline
    pipeline = dai.Pipeline()

    # Define sources and outputs
    camRgb = pipeline.create(dai.node.ColorCamera)
    spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    nnNetworkOut = pipeline.create(dai.node.XLinkOut)
    videoEnc = pipeline.create(dai.node.VideoEncoder)
    
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutNN = pipeline.create(dai.node.XLinkOut)
    xoutDepth = pipeline.create(dai.node.XLinkOut)
    xoutRgbH264 = pipeline.create(dai.node.XLinkOut)
     
    xoutRgb.setStreamName("rgb")
    xoutNN.setStreamName("detections")
    xoutDepth.setStreamName("depth")
    nnNetworkOut.setStreamName("nnNetwork")
    xoutRgbH264.setStreamName("rgbH264")
    
    # this seems to be doing nothing
    xoutRgbH264.input.setBlocking(False)
    xoutRgbH264.input.setQueueSize(1)
    xoutNN.input.setBlocking(False)
    xoutNN.input.setQueueSize(1)
    xoutDepth.input.setBlocking(False)
    xoutDepth.input.setQueueSize(1)
    nnNetworkOut.input.setBlocking(False)
    nnNetworkOut.input.setQueueSize(1)

    # Properties
    #camRgb.setPreviewSize(416, 416)
    camRgb.setPreviewSize(nn_w, nn_h)
    camRgb.setFps(30)
    # camRgb.setVideoSize(1280, 720)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    # camRgb.setPreviewKeepAspectRatio(False)
    
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    monoLeft.setCamera("left")
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    monoRight.setCamera("right")
    
    videoEnc.setDefaultProfilePreset(camRgb.getFps(), dai.VideoEncoderProperties.Profile.H264_MAIN)
    videoEnc.setKeyframeFrequency(30)
    videoEnc.setNumBFrames(0)
    #videoEnc.setRateControlMode(dai.VideoEncoderProperties.RateControlMode.VBR)
    videoEnc.setBitrateKbps(5000)
    videoEnc.setQuality(100)
    camRgb.video.link(videoEnc.input)
    videoEnc.bitstream.link(xoutRgbH264.input)
    
    # setting node configs
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
    # Align depth map to the perspective of RGB camera, on which inference is done
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(nn_w, nn_h)
    stereo.setSubpixel(False)
    # stereo.initialConfig.setConfidenceThreshold(200)
    # stereo.initialConfig.setLeftRightCheckThreshold(5)
    # stereo.setLeftRightCheck(True)
    # stereo.setExtendedDisparity(True)
    
    spatialDetectionNetwork.setBlobPath(nn_blob_path)
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
    spatialDetectionNetwork.setNumInferenceThreads(nn_inference_threads)

    # Linking
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    camRgb.preview.link(spatialDetectionNetwork.input)
    camRgb.preview.link(xoutRgb.input)

    spatialDetectionNetwork.out.link(xoutNN.input)

    stereo.depth.link(spatialDetectionNetwork.inputDepth)
    stereo.depth.link(xoutDepth.input)
    spatialDetectionNetwork.outNetwork.link(nnNetworkOut.input)

    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:

        # Print MxID, USB speed, and available cameras on the device
        print('Connected camera:', device.getProductName(), f'(MxId: {device.getDeviceInfo().getMxId()})')
        print('USB speed:', device.getUsbSpeed())
        print('IMU:', device.getConnectedIMU(), f'(Firmware: {device.getIMUFirmwareVersion()})', 'Rotation vector available' if device.getConnectedIMU() == 'BNO086' else 'Rotation vector not available (only provided by BNO086)')
        print('Connected cameras:', device.getConnectedCameras())
        print('RGB:', camRgb.getResolution(), camRgb.getFps())
        print('monoLeft:', monoLeft.getResolution(), monoLeft.getFps())
        print('monoRight:', monoRight.getResolution(), monoRight.getFps())
        print('NN:', nn_blob_path, f'{nn_w}x{nn_h}')
        
        calibData = device.readCalibration()
        
        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        previewQueue = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
        detectionNNQueue = device.getOutputQueue(name="detections", maxSize=1, blocking=False)
        depthQueue = device.getOutputQueue(name="depth", maxSize=1, blocking=False)
        networkQueue = device.getOutputQueue(name="nnNetwork", maxSize=1, blocking=False)
        rgbH264Queue = device.getOutputQueue(name="rgbH264", maxSize=1, blocking=False)

        startTime = time.monotonic()
        counter = 0
        fps = 0
        color = (255, 255, 255)
        printOutputLayersOnce = True

        #asyncio.get_event_loop().create_task(image_frame_loop(previewQueue, rgb_prev_pub, rcl_node))
        #asyncio.get_event_loop().create_task(image_frame_loop(depthQueue, depth_pub, rcl_node))
        #asyncio.get_event_loop().create_task(video_frame_loop(rgbH264Queue, rgb_h264_pub, rcl_node))
        
        # enc_loop = asyncio.new_event_loop()
        # enc_thread = asyncio.run_coroutine_threadsafe(video_frame_loop(rgbH264Queue, rgb_h264_pub, rcl_node), enc_loop)
        topic_thread = threading.Thread(target=video_frame_loop, name=f'end_loop', args=(rgbH264Queue, rgb_h264_pub, rcl_node))
        topic_thread.start()
        
        try:
            while True:
                
                rgb_intrinsics = calibData.getCameraIntrinsics(dai.CameraBoardSocket.RGB, camRgb.getResolutionWidth(), camRgb.getResolutionHeight(), keepAspectRatio=False)
                prev_intrinsics = adjust_intrinsics(sensor_res=(camRgb.getResolutionWidth(), camRgb.getResolutionHeight()),
                                                    initial_crop=(camRgb.getResolutionHeight(), camRgb.getResolutionHeight()),
                                                    output_resolution=(camRgb.getPreviewWidth(), camRgb.getPreviewHeight()),
                                                    intrinsics=rgb_intrinsics,
                                                    keepAspectRatio=True)
                msg_camera_info(rgb_intrinsics, camRgb.getResolutionWidth(), camRgb.getResolutionHeight(), 'oak_rgb_camera_optical_frame', rgb_info_pub, rcl_node)
                msg_camera_info(prev_intrinsics, camRgb.getPreviewWidth(), camRgb.getPreviewHeight(), 'oak_rgb_camera_optical_frame', prev_info_pub, rcl_node)
                fx = prev_intrinsics[0][0]
                fy = prev_intrinsics[1][1]
                # cx = rgb_intrinsics[0][2]
                # cy = rgb_intrinsics[1][2]
                
                # if not publisher_subscribed(det_3d_pub):
                #     await asyncio.sleep(0.5)
                #     continue
                
                det_3d_msg = Detection3DArray()
                det_2d_msg = Detection2DArray()
                    
                inPreview = previewQueue.get() # blocks
                inDet = detectionNNQueue.get() # blocks
                #depth = depthQueue.get() # blocks
                inNN = networkQueue.get() # blocks
                
                set_message_header('oak_rgb_camera_optical_frame', det_3d_msg)
                set_message_header('oak_rgb_camera_optical_frame', det_2d_msg)

                if printOutputLayersOnce:
                    toPrint = 'Output layer names:'
                    for ten in inNN.getAllLayerNames():
                        toPrint = f'{toPrint} {ten},'
                    print(toPrint)
                    printOutputLayersOnce = False

                prev_frame = inPreview.getCvFrame()
                #depthFrame = depth.getFrame() # depthFrame values are in millimeters

                # asyncio.get_event_loop().run_in_executor(None, msg_data_from_frame, inPreview, frame, rgb_prev_msg, rgb_prev_pub, rcl_node)
                # asyncio.get_event_loop().run_in_executor(None, msg_data_from_frame, depth, depthFrame, depth_msg, depth_pub, rcl_node)

                #depth_downscaled = depthFrame[::4]
                #max_depth = np.percentile(depth_downscaled, 99)
                #depthFrameColor = np.interp(depthFrame, (min_depth, max_depth), (0, 255)).astype(np.uint8)
                #depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

                detections = inDet.detections

                # If the frame is available, draw bounding boxes on it and show the frame
                prev_height = prev_frame.shape[0]
                prev_width  = prev_frame.shape[1]
                
                # prev_to_cam_rgb_x = 1.0 #camRgb.getResolutionWidth() / prev_width 
                # prev_to_cam_rgb_y = 1.0 #camRgb.getResolutionHeight() / prev_height
                
                for detection in detections:
                    # Denormalize bounding box to preview frame size
                    x1 = int(detection.xmin * prev_width)
                    x2 = int(detection.xmax * prev_width)
                    y1 = int(detection.ymin * prev_height)
                    y2 = int(detection.ymax * prev_height)
            
                    xSize = x2 - x1
                    ySize = y2 - y1
                    xCenter = x1 + xSize / 2.0
                    yCenter = y1 + ySize / 2.0
                    
                    # generate ros msgs 
                    res3d = Detection3D()
                    res2d = Detection2D()
                    
                    res2d.bbox.center.position.x = xCenter
                    res2d.bbox.center.position.y = yCenter
                    res2d.bbox.size_x = float(xSize)
                    res2d.bbox.size_y = float(ySize)
                    
                    p_z = detection.spatialCoordinates.z / 1000.0 # to m
                    res3d.bbox.center.position.x = detection.spatialCoordinates.x / 1000.0 # mm => m
                    res3d.bbox.center.position.y = -1.0 * detection.spatialCoordinates.y / 1000.0 # UPSIDE DOWN!
                    res3d.bbox.size.x = xSize * (p_z / fx)
                    res3d.bbox.size.y = ySize * (p_z / fy)
                    res3d.bbox.size.z = 0.01
                    
                    # hypothesis is the same for both 2d and 3d
                    hyp = ObjectHypothesisWithPose()
                    hyp.hypothesis.class_id = str(detection.label)
                    hyp.hypothesis.score = detection.confidence
                    
                    hyp.pose.pose.position.x = detection.spatialCoordinates.x / 1000.0 # mm => m
                    hyp.pose.pose.position.y = -1.0 * detection.spatialCoordinates.y / 1000.0 # UPSIDE DOWN!
                    hyp.pose.pose.position.z = detection.spatialCoordinates.z / 1000.0
                    
                    res3d.results.append(hyp)
                    res2d.results.append(hyp)
                    
                    det_3d_msg.detections.append(res3d)
                    det_2d_msg.detections.append(res2d)
                    await asyncio.sleep(0)
                            
                if rcl_node.context.ok():
                    det_3d_pub.publish(det_3d_msg)
                    det_2d_pub.publish(det_2d_msg)
                
                #await asyncio.sleep(0)
                
        except (KeyboardInterrupt, asyncio.CancelledError):
            print('Stopping async loop...')
            
        topic_thread.join()
    
    rcl_node.destroy_node()


def main():    
    rclpy.init()
    try:
        asyncio.run(async_loop())
    except (KeyboardInterrupt, asyncio.CancelledError):
        print('Stopping...')
    #rclpy.shutdown()


if __name__ == '__main__':
    main()
