from flask import Flask, redirect, url_for, render_template, session, escape, request, jsonify
from flask_mysqldb import MySQL

app = Flask(__name__)

app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'root123'
app.config['MYSQL_DB'] = 'lf'

mysql = MySQL(app)

app.secret_key = 'secretkey'

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'username' not in session:
        session['username'] = 0

    if request.method == 'POST':
        session['username'] = request.form['user_email']
        passwordform = request.form['user_password']

        if 'username' in session:
            username = session['username']
            #print str(username)
            if username == 0:
                return redirect(url_for('login'))
            
            cur = mysql.connection.cursor()
            cur.execute("select count(1) from user where user_email = '{}';".format(username))
            #return str(cur.fetchone()[0])
            rollno = cur.fetchone()[0]
            if not rollno:
                return 'Invalid Username'

            cur.execute("select user_password from user where user_email = '{}';".format(username))
            password = cur.fetchone()
            #print str(password[0])
            if passwordform == password[0]:

                #return str(session['username']+' '+password)
                return redirect(url_for('home'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username',None)
    return redirect(url_for('login'))

@app.route('/home', methods=['GET', 'POST'])
def home():
    if 'username' in session:
        username = session['username']
        #return str(username)
        if username == 0:
            return redirect(url_for('login'))

        return render_template('home.html',uname=username) 
            
    return redirect(url_for('login'))

@app.route('/lost')
def lost():
    return render_template('lost_object_background.html')

@app.route('/found')
def found():
    return render_template('check_for_found_objects.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        session['username'] = request.form['user_email']
        user_name = request.form['user_name']
        user_email = request.form['user_email']
        user_pno = request.form['user_pno']
        user_address = request.form['user_address']
        user_password = request.form['user_password']

        cur = mysql.connection.cursor()
        cur.execute("insert into user (user_name,user_pno,user_email,user_address,user_password) values('{}',{},'{}','{}','{}');".format(user_name,user_pno,user_email,user_address,user_password))
        mysql.connection.commit()

    return render_template('signup.html')

@app.route('/report', methods=['GET', 'POST'])
def report():
    return render_template('reportobject.html')



if __name__ == '__main__':
   app.run(debug=True)