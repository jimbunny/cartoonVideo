import json
import requests

douyin_parser_url = 'http://videotool.com.cn/video/superSearch'  # 第三方解析url


def parser_video(url, download_url, download_file):
    print('视频开始下载！')
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
