@echo off
call D:\isaacsim-env\Scripts\activate

set PATH=D:\isaacsim-env\Lib\site-packages\isaacsim\exts\isaacsim.ros2.bridge\bin;%PATH%
set PATH=D:\isaacsim-env\Lib\site-packages\isaacsim\exts\isaacsim.ros2.bridge\humble\lib;%PATH%

set RMW_IMPLEMENTATION=rmw_fastrtps_cpp
set ROS_DOMAIN_ID=0
set FASTRTPS_DEFAULT_PROFILES_FILE=D:\project\dronos\fastdds_wsl.xml
set ISAAC_ROS2_DISTRO=humble

cd D:\project\dronos
python launch_gui.py