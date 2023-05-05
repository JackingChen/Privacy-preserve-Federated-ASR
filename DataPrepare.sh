#!/bin/bash

SFTP_SERVER=nthuee-biic-base.synology.me
SFTP_USERNAME=jchen
SFTP_PASSWORD=jchen
SFTP_REMOTE_DIR=/ADDRESSo/
LOCAL_DIR=/home/FedASR/Addresso/
FOLDER_NAME=ADReSS-IS2020-data/
# 连接FTP服务器
sftp $SFTP_USERNAME@$SFTP_SERVER <<EOF
# 输入密码
$SFTP_PASSWORD
# 进入SFTP服务器上的目录
cd $SFTP_REMOTE_DIR
# 下载整个文件夹
get -r $FOLDER_NAME
# 关闭SFTP连接
quit
EOF
