ARG ROS_DISTRO=humble
FROM ros:$ROS_DISTRO

RUN echo "Building docker image with ROS_DISTRO=$ROS_DISTRO"

RUN apt-get update -y --fix-missing
RUN apt-get install -y ssh \
                       vim mc \
                       iputils-ping net-tools iproute2 curl \
                       pip \
                       libusb-1.0-0-dev

RUN pip install --upgrade pip

RUN pip install setuptools \
                opencv-python \
                termcolor \
                PyEventEmitter \
                depthai

# init workspace
ENV ROS_WS=/ros2_ws
RUN mkdir -p $ROS_WS/src

# fix numpy version to >= 1.25.2
RUN pip install numpy --force-reinstall

# generate entrypoint script
RUN echo '#!/bin/bash \n \
set -e \n \
\n \
# setup ros environment \n \
source "/opt/ros/'$ROS_DISTRO'/setup.bash" \n \
test -f "/ros2_ws/install/setup.bash" && source "/ros2_ws/install/setup.bash" \n \
\n \
exec "$@"' > /ros_entrypoint.sh

RUN chmod a+x /ros_entrypoint.sh

# source underlay on every login
RUN echo 'source /opt/ros/'$ROS_DISTRO'/setup.bash' >> /root/.bashrc
RUN echo 'test -f "/ros2_ws/install/setup.bash" && source "/ros2_ws/install/setup.bash"' >> /root/.bashrc

WORKDIR $ROS_WS

# install phntm bridge and agent
COPY ./ $ROS_WS/src/phntm_oak_ros2
RUN . /opt/ros/$ROS_DISTRO/setup.sh && \
    rosdep install -i --from-path src/phntm_oak_ros2 --rosdistro $ROS_DISTRO -y && \
    colcon build --symlink-install --packages-select phntm_oak_ros2

# pimp up prompt with hostame and color
RUN echo "PS1='\${debian_chroot:+(\$debian_chroot)}\\[\\033[01;35m\\]\\u@\\h\\[\\033[00m\\] \\[\\033[01;34m\\]\\w\\[\\033[00m\\] 👁️  '"  >> /root/.bashrc

WORKDIR $ROS_WS/src/phntm_oak_ros2

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD [ "bash" ]