# ใช้คำสั่งนี้นาคั้บ python scraping.py

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time

# ตั้งค่า ChromeDriver
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # รันแบบไม่เปิดหน้าต่าง
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

# เปิดเบราว์เซอร์
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

# เปิดเว็บ
url = "https://ttsview.railway.co.th/v3/search/?qType=21&qParam=3wGd4NOhq8ajnu7lGrVxzjw1e4Y7s9AARrFI&auth=befbbf5fc2c6d1c97256e07b130a942f4d1c18c3b96699ec8f91566982c25f311742750424"
driver.get(url)

# รอให้หน้าเว็บโหลด
time.sleep(5)

try:
    # หาและกดปุ่ม "แสดง 16 สถานีที่ซ่อนอยู่" โดยใช้ ID ที่ให้มา
    show_more_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.ID, "hideShowWord"))
    )

    # ทำการ scroll ไปยังปุ่มเพื่อให้มั่นใจว่ามองเห็นได้
    driver.execute_script("arguments[0].scrollIntoView();", show_more_button)
    time.sleep(1)

    # คลิกที่ปุ่ม
    show_more_button.click()
    time.sleep(2)  # รอให้ข้อมูลโหลด

    print("✅ คลิกปุ่ม 'แสดง 16 สถานีที่ซ่อนอยู่' สำเร็จ!")
except Exception as e:
    print(f"❌ ไม่สามารถคลิกปุ่มได้: {e}")

    # แสดงรายละเอียดของหน้าเว็บเพื่อการแก้ไขปัญหา
    print("\nHTML ของพื้นที่ที่น่าจะมีปุ่ม:")
    try:
        section_html = driver.find_element(By.TAG_NAME, "body").get_attribute("innerHTML")
        print(section_html[:500] + "...")  # แสดงเฉพาะส่วนต้น
    except:
        print("ไม่สามารถดึง HTML ได้")

# หา element ที่มีข้อมูลที่ต้องการ
try:
    # รอให้ elements ปรากฏหลังจากคลิกปุ่ม
    WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.CLASS_NAME, "realTime"))
    )

    elements = driver.find_elements(By.CLASS_NAME, "realTime")

    # แสดงข้อมูลที่ดึงมา
    if elements:
        for index, element in enumerate(elements, start=1):
            print(f"ข้อมูล {index}: {element.text}")
    else:
        print("ไม่พบข้อมูลที่มี class 'realTime'")
except Exception as e:
    print(f"❌ เกิดข้อผิดพลาดในการดึงข้อมูล: {e}")

# ถ่ายภาพหน้าจอเพื่อตรวจสอบ (optional)
try:
    driver.save_screenshot("debug_screenshot.png")
    print("✅ บันทึกภาพหน้าจอสำหรับตรวจสอบแล้ว")
except:
    print("❌ ไม่สามารถบันทึกภาพหน้าจอได้")

# ปิดเบราว์เซอร์
driver.quit()