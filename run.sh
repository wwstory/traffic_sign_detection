#!/bin/sh

if [ ! -f '/tmp/test.jpg' ]; then
    echo '测试图片不存在于 /tmp/test.jpg'
    exit 1
fi

python3 test.py