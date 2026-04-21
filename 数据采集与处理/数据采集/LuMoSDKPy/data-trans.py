import LuMoSDKClient as LuMoSDKClient
import socket

ip = "127.0.0.1"
target_ip = "172.16.0.1"

LuMoSDKClient.Init()
LuMoSDKClient.Connnect(ip)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

count = 0
while True:
    print(f"\ncount: {count}")
    count += 1
    frame = LuMoSDKClient.ReceiveData(0)    # 0 :阻塞接收 1：非阻塞接收
    if frame is None:
        continue      

    FrameID = frame.FrameId
    TimeStamp = frame.TimeStamp
    uCameraSyncTime = frame.uCameraSyncTime
    uBroadcastTime = frame.uBroadcastTime        
    print(f"frame_ID: {FrameID} | Current_Time_Stamp: {TimeStamp} | Camera_Sync_Time: {uCameraSyncTime} | Data_Broadcast_Time: {uBroadcastTime}")

    markers = frame.markers
    data = f"{FrameID},{len(markers)},{TimeStamp},{uCameraSyncTime},{uBroadcastTime},"
    for i, marker in enumerate(markers):
        print(f"    marker_i: {i}, X: {marker.X}, Y: {marker.Y}, Z: {marker.Z}")
        data += f"{marker.X},{marker.Y},{marker.Z},"

    rigids = frame.rigidBodys
    for i, rigid in enumerate(rigids):
        print(f"    rigid_i: {i}, X: {rigid.X}, Y: {rigid.Y}, Z: {rigid.Z}, qx: {rigid.qx}, qy: {rigid.qy}, qz: {rigid.qz}, qw: {rigid.qw},")
        data += f"{rigid.X},{rigid.Y},{rigid.Z},{rigid.qx},{rigid.qy},{rigid.qz},{rigid.qw},"
        
    sock.sendto(data.encode('utf-8'), (target_ip, 10101))