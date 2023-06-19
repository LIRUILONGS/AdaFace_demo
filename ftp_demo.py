#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   ftp_demo.py
@Time    :   2023/06/18 22:33:24
@Author  :   Li Ruilong
@Version :   1.0
@Contact :   liruilonger@gmail.com
@Desc    :   FTP  服务器
"""

# here put the import lib
from ftpdlib import FTPServer

# 定义一个处理器类
class MyHandler:
    def on_connect(self, ftp_handler):
        print("New connection:", ftp_handler.remote_ip)
    def on_disconnect(self, ftp_handler):
        print("Disconnected:", ftp_handler.remote_ip)
    def on_file_sent(self, ftp_handler, file):
        print("File sent:", file)
    def on_file_received(self, ftp_handler, file):
        print("File received:", file)

# 创建一个 FTP 服务器实例
address = ("", 21)  # 绑定本机 IP 和端口 21
handler = MyHandler()
server = FTPServer(address, handler)

# 启动服务器
server.serve_forever()
