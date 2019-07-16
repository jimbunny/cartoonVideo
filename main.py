# -*- coding: utf-8 -*-
# @Time    : 2019/7/16 13:56
# @Author  : jim
# @Email   : jingtongyu@126.com
# @File    : main.py(视频合成动画)


import requests
import json

douyin_parser_url = 'http://videotool.com.cn/video/superSearch'  # 第三方解析url
download_url = 'http://videotool.com.cn/video/loadNative?videoUrl='  # 解析后下载地址
download_file = "D:\\pythonpractice\\download\\test.mp4"


def parser_video(url, download_url):
    headers = {
        "Content-Type": "application/json",
    }
    data = json.dumps({"url": url})
    r = requests.post(douyin_parser_url, headers=headers, data=data)
    if r.status_code == 200:
        if json.loads(r.text).get("code") == 0:
            parser_url = json.loads(r.text).get("content")
            download_url = download_url + parser_url
            r = requests.get(download_url)  # create HTTP response object
            with open(download_file, 'wb') as f:
                f.write(r.content)
            print('视频下载成功！目录：' + download_file)
        else:
            print('视频解析失败' + r.text.msg)
    else:
        print('视频解析失败！请求第三方平台失败！')


if __name__ == '__main__':
    url = 'http://v.douyin.com/B8YsRY/'
    parser_video(url, download_url)
