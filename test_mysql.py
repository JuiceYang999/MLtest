import pymysql
import sys

# --- 修改为您的配置 ---
DB_HOST = "127.0.0.1"  # 或您的 MySQL 服务器 IP
DB_PORT = 3306
DB_USER = "root"
DB_PASS = "cps@CPS123"
DB_NAME = "mqtt_data"
# ---------------------

try:
    connection = pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )
    print(f"✅ 成功连接到 MySQL (Host: {DB_HOST})")

    with connection.cursor() as cursor:
        # 执行一个简单的查询
        cursor.execute("SHOW TABLES;")
        tables = cursor.fetchall()
        print(f"👉 在 '{DB_NAME}' 数据库中找到的表:")
        if tables:
            for i, table in enumerate(tables):
                # PyMySQL 1.1+ cursors return dicts
                # The key is 'Tables_in_{db_name}'
                print(f"  {i+1}. {list(table.values())[0]}")
        else:
            print("  (未找到表)")

except pymysql.Error as e:
    print(f"❌ MySQL 连接失败: {e}", file=sys.stderr)
    sys.exit(1)
finally:
    if 'connection' in locals() and connection:
        connection.close()
        print("\n✅ MySQL 连接已关闭。")