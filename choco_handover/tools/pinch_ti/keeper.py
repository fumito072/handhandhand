"""常駐接続。ブリッジは最後のクライアントが切れると指の保持を解除するので、これを動かしておく (受信するだけ)。"""
import socket, struct, time
while True:
    try:
        s = socket.create_connection(("192.168.123.164", 8891), timeout=5); s.settimeout(2.0); buf = b""
        while True:
            c = s.recv(1 << 18)
            if not c: break
            buf += c
            while len(buf) >= 5:
                ln, mt = struct.unpack(">IB", buf[:5])
                if len(buf) < 4 + ln: break
                buf = buf[4 + ln:]
    except Exception: time.sleep(1.0)
