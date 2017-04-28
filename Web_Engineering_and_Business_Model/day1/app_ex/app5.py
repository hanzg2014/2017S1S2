# coding: utf-8

from flask import Flask, render_template, request
from flask_mysqldb import MySQL

app = Flask(__name__)

mysql = MySQL(app)

# # MySQL configurations
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'root'
app.config['MYSQL_DB'] = 'world'
# app.config['MYSQL_DATABASE_HOST'] = 'localhost'
# mysql.init_app(app)
# conn = mysql.connect()
# cursor = conn.cursor()

@app.route("/mysqluser")
def mysqluser():
    print ('cur')
    cur = mysql.connection.cursor()
    cur.execute('''SELECT user, host FROM mysql.user''')
    rv = cur.fetchall()
    return str(rv)
    # return render_template('index3.html')

@app.route("/authenticate")
def Authenticate():
    # http://127.0.0.1:5000/authenticate?UserName=root&Password=root
    username = request.args.get('UserName')
    password = request.args.get('Password')
    cursor = mysql.connection.cursor()
    cursor.execute("SELECT * from User where Username='" + username + "' and Password='" + password + "'")
    data = cursor.fetchone()
    if data is None:
     return "Username or Password is wrong"
    else:
     return "Logged in successfully"

#if __name__ == "__main__":
    #app.run(debug=True, host="0.0.0.0")
