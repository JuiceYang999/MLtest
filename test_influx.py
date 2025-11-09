from influxdb_client import InfluxDBClient
from influxdb_client.client.exceptions import InfluxDBError
import sys

# --- 修改为您的 InfluxDB 2.x 配置 ---
INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "7ucc4S8rrzwu85NA5nUYb_CNG7C-03Rbuyf2A85A5leATuxcPH_UlFvrCNXSGQtxvZQTuY_C6O7BUWNg4oIH-g==" # 你的本地 Token
INFLUX_ORG = "XJTU"
# ------------------------------------

try:
    client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)

    # 检查连接 (Health Check)
    health = client.health()
    if health.status != "pass":
        raise Exception(f"InfluxDB health check failed: {health.message}")

    print(f"✅ 成功连接到 InfluxDB (URL: {INFLUX_URL}, Org: {INFLUX_ORG})")

    # 尝试列出所有的 Buckets
    buckets_api = client.buckets_api()
    buckets = buckets_api.find_buckets()

    print("👉 找到的 Buckets:")
    if buckets.buckets:
        for i, bucket in enumerate(buckets.buckets):
            print(f"  {i+1}. {bucket.name} (ID: {bucket.id})")
    else:
        print("  (未找到 Buckets)")

except InfluxDBError as e:
    print(f"❌ InfluxDB API 错误: {e}", file=sys.stderr)
    if e.response.status == 401:
        print("   (提示: 401 Unauthorized - 请检查您的 Token 是否正确且具有所需权限。)", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"❌ InfluxDB 连接失败: {e}", file=sys.stderr)
    sys.exit(1)
finally:
    if 'client' in locals() and client:
        client.close()
        print("\n✅ InfluxDB 连接已关闭。")