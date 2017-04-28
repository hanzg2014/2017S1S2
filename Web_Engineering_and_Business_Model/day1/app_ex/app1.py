from flask import Flask, render_template

print ("test1 works!")

app = Flask(__name__)

@app.route("/")
def index():
    # templates/index.html を参照
    # message （変数）に"Hello"と代入し、templateに反映
    return render_template('index1.html', message="Hello")

#if __name__ == "__main__":
    #app.run(debug=True)
