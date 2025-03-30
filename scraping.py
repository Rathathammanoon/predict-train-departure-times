from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
from datetime import datetime
import re

options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--window-size=1920,1080")

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

url = "https://ttsview.railway.co.th/v3/search/?qType=21&qParam=3wGd4NOhq8ajnu7lGrVxzjw1e4Y7s9AARrFI&auth=befbbf5fc2c6d1c97256e07b130a942f4d1c18c3b96699ec8f91566982c25f311742750424"
driver.get(url)

print("รอให้หน้าเว็บโหลด...")
time.sleep(1)  # เพิ่มเวลารอเป็น 1 วินาที

train_data = []

try:
    driver.execute_script("showPassStation(16);")
    print("✅ กดปุ่มแสดงข้อมูลเพิ่มเติม สำเร็จ!")
    print("รอให้ข้อมูลโหลด...")
    # time.sleep(1)  # เพิ่มเวลารอเป็น 1 วินาที

    train_name = "ไม่พบชื่อรถไฟ"
    try:
        train_name_element = driver.find_element(By.CLASS_NAME, "trainLink")
        train_name = train_name_element.text
        print(f"ชื่อรถไฟ (วิธีที่ 1): {train_name}")
    except Exception as e:
        print(f"ไม่สามารถดึงชื่อรถไฟด้วยวิธีที่ 1")

        try:
            train_name_element = driver.find_element(By.XPATH, "//div[contains(@class, 'train-name')]")
            train_name = train_name_element.text
            print(f"ชื่อรถไฟ (วิธีที่ 2): {train_name}")
        except Exception as e:
            print(f"ไม่สามารถดึงชื่อรถไฟด้วยวิธีที่ 2")

    print("\nค้นหาข้อมูลเวลา:")
    real_time_elements = driver.find_elements(By.CLASS_NAME, "realTime")
    print(f"พบข้อมูลเวลาจริง: {len(real_time_elements)}")

    train_date = datetime.now().strftime("%d/%m/%Y")  # ใช้วันที่ปัจจุบันเป็นค่าเริ่มต้น
    try:
        date_element = driver.find_element(By.XPATH, "//div[contains(@class, 'date')]")
        train_date = date_element.text
    except:
        print("ไม่พบข้อมูลวันที่")

    # ดึงข้อมูลทั้งหมดจาก row ของตาราง
    station_rows = driver.find_elements(By.XPATH, "//div[contains(@class, 'station-row')]")
    if not station_rows:
        station_rows = driver.find_elements(By.XPATH, "//tr[contains(@class, 'station')]")

    if len(station_rows) > 0:
        print(f"พบข้อมูลแถวสถานีทั้งหมด {len(station_rows)} รายการ")

        for row in station_rows:
            try:
                station = row.find_element(By.XPATH, ".//*[contains(@class, 'station-name')]").text
            except:
                try:
                    station = row.find_element(By.XPATH, ".//td[1]").text
                except:
                    station = "ไม่พบชื่อสถานี"

            try:
                scheduled_time = row.find_element(By.XPATH, ".//*[contains(@class, 'scheduledTime')]").text
            except:
                try:
                    scheduled_time = row.find_element(By.XPATH, ".//td[2]").text
                except:
                    scheduled_time = "-"

            try:
                real_time_elem = row.find_element(By.XPATH, ".//*[contains(@class, 'realTime')]")
                real_time = real_time_elem.text
            except:
                try:
                    real_time = row.find_element(By.XPATH, ".//td[3]").text
                except:
                    real_time = "-"

            time_diff = ""
            try:
                if real_time != "-":
                    match = re.search(r'([+-]\d+)', real_time)
                    if match:
                        time_diff = match.group(1)
            except:
                time_diff = ""

            train_data.append({
                'รถไฟ': train_name,
                'วันที่': train_date,
                'สถานี': station,
                'เวลาตามตาราง': scheduled_time,
                'เวลาจริง': real_time,
                'ความแตกต่าง (นาที)': time_diff
            })
    else:
        # ถ้าไม่พบแถวสถานี ให้พยายามแยกดึงข้อมูลจากกลุ่มเวลาจริงที่พบ
        if real_time_elements:
            print(f"ไม่พบแถวสถานี แต่พบข้อมูลเวลาจริง {len(real_time_elements)} รายการ")
            for i, real_time_elem in enumerate(real_time_elements):
                train_data.append({
                    'รถไฟ': train_name,
                    'วันที่': train_date,
                    'สถานี': f"สถานีที่ {i + 1}",
                    'เวลาตามตาราง': "-",
                    'เวลาจริง': real_time_elem.text,
                    'ความแตกต่าง (นาที)': ""
                })
        else:
            print("ไม่พบข้อมูลสถานีหรือเวลาเลย")

    if train_data:
        df = pd.DataFrame(train_data)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_filename = f"train_data_{timestamp}.xlsx"

        df.to_excel(excel_filename, index=False)
        print(f"✅ บันทึกข้อมูลลงไฟล์ {excel_filename} เรียบร้อยแล้ว")
    else:
        print("❌ ไม่มีข้อมูลสำหรับบันทึกลง Excel")

except Exception as e:
    print(f"❌ เกิดข้อผิดพลาด: {e}")

driver.quit()

if train_data:
    print(f"\nดึงข้อมูลได้ทั้งหมด {len(train_data)} รายการ")
else:
    print("\nไม่มีข้อมูลที่ดึงได้")