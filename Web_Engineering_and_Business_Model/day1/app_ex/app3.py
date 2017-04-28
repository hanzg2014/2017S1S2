from flask import Flask, render_template, request

print ("test2 works!")

app = Flask(__name__)

@app.route("/")
def index():
    # templates/index.html を参照
    # message （変数）に"Hello"と代入し、templateに反映
    return render_template('index3.html')

@app.route("/hello")
def hello():
    # request.argsにクエリパラメータが含まれている
    val = request.args.get("msg", "Not defined")
    return 'Hello World '  + val

@app.route('/get_type', methods=['POST'])
def get_type():
    # request.formにPOSTデータがある
    goods = request.form["goods"]
    return 'You get ' + goods

@app.route('/get_price/<goodsid>')
def get_price(goodsid):
    goods = [('Apple',300), ('Banana',100),('Peach',350)]
    # price = goods[goodsid].key
    price = 70
        # return 'You should pay %d yen' % price
    return 'You should pay'

#if __name__ == "__main__":
    #app.run(debug=True, host="0.0.0.0")
