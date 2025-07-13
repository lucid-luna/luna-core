# ====================================================================
#  File: utils/osc_sender.py
# ====================================================================
"""
L.U.N.A. <-> Unity OSC 송신 모듈
"""

from pythonosc.udp_client import SimpleUDPClient

class OSCSender:
    def __init__(self, ip: str = "127.0.0.1", port: int = 9000):
        self.client = SimpleUDPClient(ip, port)
        
    def send_emotion(self, emotion: str):
        self.client.send_message("/luna/emotion", emotion)
        print(f"[L.U.N.A. OSC] 감정 전송: {emotion}")