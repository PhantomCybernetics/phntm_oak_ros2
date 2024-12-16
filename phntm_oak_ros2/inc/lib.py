import time
import math
import depthai as dai
import numpy as np
import array
import asyncio
from termcolor import colored as c
import av
import fractions

from sensor_msgs.msg import Image
from ffmpeg_image_transport_msgs.msg import FFMPEGPacket
from sensor_msgs.msg import CameraInfo

from typing import List, Tuple
from copy import deepcopy


NS_TO_SEC = 1000000000
# VIDEO_PTIME = 1 / 30  # 30fps
SRC_VIDEO_TIME_BASE = fractions.Fraction(1, NS_TO_SEC)

num_copy_samples = 0
time_copy_total = 0.0
time_send_total = 0.0

last_pub_sub_states = {}
def publisher_subscribed(pub) -> bool:
    global last_pub_sub_states
    try:
        has_subs = pub.get_subscription_count() > 0
        log = not pub.topic_name in last_pub_sub_states.keys() \
            or last_pub_sub_states[pub.topic_name] != has_subs
        if log:
            print(c(f'Publisher {pub.topic_name} {"subscribed" if has_subs else ("unsubscribed" if pub.topic_name in last_pub_sub_states.keys() else "not subscribed")}', 'green' if has_subs else 'cyan'))
        last_pub_sub_states[pub.topic_name] = has_subs
        return has_subs
    except:
        return False


def set_message_header(frame, msg):
    time_nanosec:int = time.time_ns()
    msg.header.stamp.sec = math.floor(time_nanosec / 1000000000)
    msg.header.stamp.nanosec = time_nanosec % 1000000000
    msg.header.frame_id = frame


async def image_frame_loop(q, pub, rcl_node):
    try:
    
        while True:
            
            if not publisher_subscribed(pub):
                await asyncio.sleep(0.5)
                continue
            
            if q.has():
                buf = q.get()
                frame = buf.getCvFrame()
                msg = Image()
                set_message_header('', msg)
                
                asyncio.get_event_loop().run_in_executor(None, msg_data_from_frame, buf, frame, msg, pub, rcl_node)
            
            await asyncio.sleep(0.001)
            
    except (KeyboardInterrupt, asyncio.CancelledError):
        print('Stopping frame loop...')

def video_frame_loop(q, pub, rcl_node):
    try:
        while True:
            
            # if not publisher_subscribed(pub):
            #     sleep()
            #     continue
            
            # if not q.has():
            #     await asyncio.sleep(0.001)
            #     continue
            
            buf = q.get() # blocks
            frame_bytes = buf.getData()
            
            # THIS DOES NOTHING
            # frame = dai.EncodedFrame()
            # frame.setFrameType(dai.EncodedFrame.FrameType.Unknown)
            # frame.setData(frame_bytes)
            # keyframe = frame.getFrameType() == dai.EncodedFrame.FrameType.I
            # print (f'Got video frame {len(frame_bytes)}B {frame.getFrameType()}')
            keyframe = True
            
            msg = FFMPEGPacket()
            set_message_header('', msg)
            msg.width = buf.getWidth()
            msg.height = buf.getHeight()
            msg.encoding = 'h.264'
            msg.pts = 0 # ns # 'uint64',
            msg.flags = 1 if keyframe else 0 # 'uint8',
            msg.is_bigendian = False            
            frame_contiguous = np.ascontiguousarray(frame_bytes, dtype=np.uint8)
            msg.data = array.array('B', frame_contiguous.tobytes())  #array.array('B', )
            # asyncio.get_event_loop().run_in_executor(None, msg_data_from_frame, buf, frame, msg, pub, rcl_node)
            
            if rcl_node.context.ok():
                pub.publish(msg)
                
            # await asyncio.sleep(0.001)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print('Stopping video thread...')
            

def msg_data_from_frame(buf, frame, msg, pub, rcl_node):
    msg.width = buf.getWidth()
    msg.height = buf.getHeight()
    # time_start = time.time()
    #print(f'Frame {inPreview.getWidth()}x{inPreview.getHeight()} type={inPreview.getType()}')
    match buf.getType():
        case dai.RawImgFrame.Type.RAW16:
            msg.encoding = '16UC1'
            msg.is_bigendian = False
            frame_contiguous = np.ascontiguousarray(frame, dtype=np.uint16)
            msg.data = array.array('B', frame_contiguous.tobytes())  #array.array('B', )
        case dai.RawImgFrame.Type.BGR888p:
            msg.encoding = 'bgr8'
            msg.is_bigendian = True
            frame_contiguous = np.ascontiguousarray(frame, dtype=np.uint8)
            msg.data = array.array('B', frame_contiguous.tobytes())  #array.array('B', )
        case _:
            print(f'Ignoring frame of unsuppored {buf.getType()}')
            return

    # global num_copy_samples, time_copy_total, time_send_total
    # time_copy = time.time() - time_start
    # time_copy_total += time_copy
        
    if rcl_node.context.ok():
        # time_start = time.time()
        pub.publish(msg)
        # time_send = time.time() - time_start
        # time_send_total += time_send
    
    # num_copy_samples += 1
    # if num_copy_samples == 100:
    #     print(f'{num_copy_samples} samples; avg_copy={time_copy_total/num_copy_samples:.5f}s; avg_send={time_send_total/num_copy_samples:.5f}')
    #     time_copy_total = 0.0
    #     time_send_total = 0.0
    #     num_copy_samples = 0


def msg_camera_info(intrinsics, width, height, frame, pub, rcl_node):
    

    msg = CameraInfo()
    set_message_header(frame, msg)
    
    msg.width = width
    msg.height = height
    
    msg.k = np.ndarray(shape=(9,))
    msg.k[0] = intrinsics[0][0]
    msg.k[1] = intrinsics[0][1]
    msg.k[2] = intrinsics[0][2]
    msg.k[3] = intrinsics[1][0]
    msg.k[4] = intrinsics[1][1]
    msg.k[5] = intrinsics[1][2]
    msg.k[6] = intrinsics[2][0]
    msg.k[7] = intrinsics[2][1]
    msg.k[8] = intrinsics[2][2]
    
    if rcl_node.context.ok():
        pub.publish(msg)
    
    return intrinsics

# from https://github.com/Taiga-Robotics/luxonis_support/blob/main/scripts/oak_d_node.py#L121
# mentioned https://discuss.luxonis.com/d/2235-intrinsics-of-preview/4
def adjust_intrinsics(sensor_res:Tuple[int, int], initial_crop:Tuple[int, int], output_resolution:Tuple[int, int], 
                      intrinsics: List[List[float]], keepAspectRatio:bool = True) -> List[List[float]]:
    """
    sensor_res: resolution of sensor for which the intrinsics will be supplied
    initial_crop: cropping of sensor res before scaling
    output resolution: scaling of sensor and initial crop
    intrinsics: 3x3 [[f_x, 0, c_x],[0, f_y, c_y], [0,0,1]]
    keepaspectratio: if true then the smallest scaling factor will be applied from (width, height) and the other axis is assumed to be scaled and cropped to output resolution
    x is width, y is height. 
    """
    intrinsics = deepcopy(intrinsics)
    # adjust optical centre due to initial crop

    # print(intrinsics)   # debugging

    if initial_crop:
        # TODO review this for off by one errors.
        xoff = (sensor_res[0] - initial_crop[0])/2
        yoff = (sensor_res[1] - initial_crop[1])/2
        intrinsics[0][2]-= xoff
        intrinsics[1][2]-= yoff
        #update sensor res for subsequent math
        sensor_res=initial_crop

    # print(intrinsics)   # debugging

    # scale centre and f
    if output_resolution:
        scalex = output_resolution[0]/sensor_res[0]
        scaley = output_resolution[1]/sensor_res[1]
        cropaxis=0
        cropoffset=0.0
        if keepAspectRatio:
            if scalex >= scaley:
                scaley=scalex
                cropaxis = 1
            if scaley > scalex:
                scalex=scaley
                cropaxis = 0
            
            # calculate adjustment to c_(x||y) to be applied after scaling
            cropoffset = (sensor_res[cropaxis]*scalex - output_resolution[cropaxis])/2.0

        intrinsics[0][0]*=scalex
        intrinsics[0][2]*=scalex
        intrinsics[1][1]*=scaley
        intrinsics[1][2]*=scaley

        # TODO review this for off by one errors.
        intrinsics[cropaxis][2]-=cropoffset
    
    # print(intrinsics)   # debugging
    # print("===================")

    return intrinsics