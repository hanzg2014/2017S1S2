from flask import Flask

print ("test0 works!")

app = Flask(__name__)

# iLectの場合はルートを"/"でなく"/a/"とする
@app.route("/")
def index():

    return "Hello World!"

#if __name__ == "__main__":
    #iLectの場合は[host="0.0.0.0"]をプロパティに追加
    #app.run(debug=True, host="0.0.0.0")
