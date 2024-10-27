import time
import math
import depthai as dai
import numpy as np
import array

def set_message_header(frame, msg):
    time_nanosec:int = time.time_ns()
    msg.header.stamp.sec = math.floor(time_nanosec / 1000000000)
    msg.header.stamp.nanosec = time_nanosec % 1000000000
    msg.header.frame_id = frame
    
def msg_data_from_frame(buf, frame, msg):
    msg.width = buf.getWidth()
    msg.height = buf.getHeight()
    #print(f'Frame {inPreview.getWidth()}x{inPreview.getHeight()} type={inPreview.getType()}')
    match buf.getType():
        case dai.RawImgFrame.Type.RAW16:
            msg.encoding = '16UC1'
            msg.is_bigendian = False
            frame_contiguous = np.ascontiguousarray(frame, dtype=np.uint16)
            msg.data = array.array('B', frame_contiguous.ravel().tobytes())  #array.array('B', )
        case dai.RawImgFrame.Type.BGR888p:
            msg.encoding = 'bgr8'
            msg.is_bigendian = True
            frame_contiguous = np.ascontiguousarray(frame, dtype=np.uint8)
            msg.data = array.array('B', frame_contiguous.tobytes())  #array.array('B', )