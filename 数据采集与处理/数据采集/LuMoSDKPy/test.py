import socket

target_ip = "172.20.10.3"
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

while True:
    sock.sendto(b"Hello", (target_ip, 10101))
    print("1")