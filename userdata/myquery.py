import mysql.connector

def connection():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='userdata'
    )
def userdatainsertion(n,e,p):
    con=connection()
    cursor=con.cursor()
    qry='INSERT INTO mytable(name,email,phone) VALUES (%s,%s,%s)'
    value=(n,e,p)
    cursor.execute(qry,value)
    con.commit()
    cursor.close()
    con.close()


def userdatafetch():
    con=connection()
    cursor=con.cursor()
    qry='select * from mytable'
    cursor.execute(qry)
    data=cursor.fetchall()
    cursor.close()
    con.close()
    return data