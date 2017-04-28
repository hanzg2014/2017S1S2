# coding: utf-8

from flask import Flask, render_template, request

app = Flask(__name__)
# (__name__) 他からimportされたとき 拡張子なしのファイル名（モジュール名）という値を格納しています。

@app.route("/")
def hello():
    return render_template('index2.html')

@app.route("/echo")
def echo():
    return "You said: " + request.args.get('text', '')

@app.route('/getter', methods=['GET'])
def getter():
    #複数の値をリストで受け取れる
    data = request.args.getlist('data')
    return ','.join(data)

# methodsにPOSTを指定すると、POSTリクエストを受けられる
@app.route('/post_request', methods=['POST'])
def post_request():
    # request.formにPOSTデータがある
    username = request.form["username"]
    return 'Thank you ' + username

@app.route('/post/<postid>')
def post(postid):
    return 'Thanks post: id = %s' % postid

#if __name__ == "__main__":
    #app.run(debug=True, host="0.0.0.0")
