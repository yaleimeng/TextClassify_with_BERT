# -*- coding: utf-8 -*-
'''
@author: yaleimeng@sina.com
@license: (C) Copyright 2019
@desc: 项目描述。
@DateTime: Created on 2019/7/22, at 下午 05:07 by PyCharm
'''

from sanic import Sanic
from sanic.response import json as Rjson
from predict_GPU import Bert_Class

app = Sanic()
my = Bert_Class()


@app.route("/", methods=['GET', 'POST'])
async def home(request):
    # 1，首先要从HTTP请求获取用户的字符串
    dict1 = {'tips': '请用POST方法，传递“用户id、question”字段'}
    if request.method == 'GET':
        user, key_str = request.args.get('user_id'), request.args.get('question')
    elif request.method == 'POST':
        for k, v in request.form.items():
            dict1[k] = v  # 最关心的问题字段是keyword
        user, key_str = request.form.get('user_id'), request.form.get('question')
    else:
        return Rjson(dict1)
    if not key_str or not user:  # 如果有空的字段，返回警告信息。
        return Rjson(dict1)

    # 2，调用自身的功能，执行搜索引擎爬虫
    dict1.pop('tips')
    dict1['Type'] = my.yuce(key_str)
    return Rjson(dict1)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5400)
