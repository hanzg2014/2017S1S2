from flask import Flask, render_template, request

print ("test2 works!")

app = Flask(__name__)


@app.route("/")
def index():
    # templates/index.html を参照
    # message （変数）に"Hello"と代入し、templateに反映
    return render_template('index2.html', message="Hello")

@app.route("/hello")
def hello():
    # request.argsにクエリパラメータが含まれている
    name = request.args.get("name", "Mr.Who")
    return 'Hello '  + name

@app.route("/hello2")
def hello2():
    # request.argsにクエリパラメータが含まれている
    name = request.args.get("name", "Mr.Who")
    msg = request.args.get("msg", "No Message")
    return 'Hello %s san! %s' % (name, msg)

#if __name__ == "__main__":
    #app.run(debug=True, host="0.0.0.0")
