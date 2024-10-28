import time
import math
import depthai as dai
import numpy as np
import array
import asyncio

import av
import fractions

from sensor_msgs.msg import Image
from ffmpeg_image_transport_msgs.msg import FFMPEGPacket

NS_TO_SEC = 1000000000
# VIDEO_PTIME = 1 / 30  # 30fps
SRC_VIDEO_TIME_BASE = fractions.Fraction(1, NS_TO_SEC)

num_copy_samples = 0
time_copy_total = 0.0
time_send_total = 0.0

def set_message_header(frame, msg):
    time_nanosec:int = time.time_ns()
    msg.header.stamp.sec = math.floor(time_nanosec / 1000000000)
    msg.header.stamp.nanosec = time_nanosec % 1000000000
    msg.header.frame_id = frame


async def image_frame_loop(q, pub, rcl_node):
    try:
    
        while True:
            
            if q.has():
                buf = q.get()
                frame = buf.getCvFrame()
                msg = Image()
                set_message_header('', msg)
                
                asyncio.get_event_loop().run_in_executor(None, msg_data_from_frame, buf, frame, msg, pub, rcl_node)
            
            await asyncio.sleep(0.001)
            
    except (KeyboardInterrupt, asyncio.CancelledError):
        print('Stopping frame loop...')

async def video_frame_loop(q, pub, rcl_node):
    try:
        
        while True:
            if q.has():
                buf = q.get()
                frame_bytes = buf.getData()
                frame = dai.EncodedFrame()
                frame.setFrameType(dai.EncodedFrame.FrameType.Unknown)
                frame.setData(frame_bytes)
                
                keyframe = frame.getFrameType() == dai.EncodedFrame.FrameType.I
                #print (f'Got video frame {len(frame_bytes)}B {frame.getFrameType()}')
                
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
                
            await asyncio.sleep(0.001)
            
    except (KeyboardInterrupt, asyncio.CancelledError):
        print('Stopping frame loop...')

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