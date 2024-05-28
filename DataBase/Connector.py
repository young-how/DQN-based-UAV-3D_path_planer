import os

import mysql.connector

from BaseClass.CalMod import XML2Dict


class MySQLConnector:
    def __init__(self, param):
        #初始化数据库
        if param == None:
            DB_xml_path = os.path.normpath("./config/DB.xml")
            DB_config_dict = XML2Dict(DB_xml_path)
            param = DB_config_dict.get('DB')
        self.host = param.get("host")
        self.user = param.get("user")
        self.password = param.get("password")
        self.database = param.get("database")
        self.table_list=param.get("tables").get("table")  #获取表集合
        self.connection =None
        self.cursor =None
        self.connect()

    def connect(self):
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            self.cursor = self.connection.cursor()
            print("Connected to MySQL database")
        except mysql.connector.Error as err:
            print(f"Error: {err}")

    def disconnect(self):
        try:
            if self.connection.is_connected():
                self.cursor.close()
                self.connection.close()
                print("Disconnected from MySQL database")
        except mysql.connector.Error as err:
            pass
            #print(f"Error: {err}")

    def insert_data(self, table, data):
        try:
            columns = ', '.join(data.keys())
            values = ', '.join(['%s'] * len(data))
            sql = f"INSERT INTO {table} ({columns}) VALUES ({values})"
            #print(sql)
            self.cursor.execute(sql, list(data.values()))
            self.connection.commit()
            #print("Data inserted successfully")
        except mysql.connector.Error as err:
            self.connection.rollback()
            #print(f"Error: {err}")