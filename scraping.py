from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
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
    # วิธีที่ 1: เรียกฟังก์ชัน JavaScript โดยตรง
    driver.execute_script("showPassStation(16);")
    time.sleep(2)  # รอให้ข้อมูลโหลด
    print("✅ เรียกฟังก์ชัน showPassStation(16) สำเร็จ!")
except Exception as e:
    print(f"❌ ไม่สามารถเรียกฟังก์ชันได้: {e}")

    # ถ้าวิธีที่ 1 ไม่สำเร็จ ให้ลองวิธีที่ 2
    try:
        print("กำลังลองวิธีที่ 2...")
        # ใช้ JavaScript เพื่อกำจัดอีเลเมนต์ที่บังและคลิกที่ปุ่ม
        driver.execute_script("""
            // หาอีเลเมนต์ที่บังอยู่และทำให้มองไม่เห็นชั่วคราว
            var overlays = document.getElementsByClassName('ant-card-body');
            for (var i = 0; i < overlays.length; i++) {
                overlays[i].style.pointerEvents = 'none';
            }

            // คลิกที่ปุ่มโดยตรง
            document.getElementById('hideShowWord').click();
        """)
        time.sleep(2)
        print("✅ คลิกปุ่มด้วย JavaScript สำเร็จ!")
    except Exception as e:
        print(f"❌ วิธีที่ 2 ไม่สำเร็จ: {e}")

        # ถ้าวิธีที่ 2 ไม่สำเร็จ ให้ลองวิธีที่ 3
        try:
            print("กำลังลองวิธีที่ 3...")
            # ใช้ Actions และการกด Tab เพื่อนำทางไปยังปุ่ม
            actions = ActionChains(driver)
            actions.send_keys(Keys.TAB * 10)  # กด Tab หลายครั้งเพื่อนำทางไปยังปุ่ม
            actions.send_keys(Keys.ENTER)  # กด Enter เพื่อเลือก
            actions.perform()
            time.sleep(2)
            print("✅ นำทางด้วยแป้นพิมพ์และกด Enter สำเร็จ!")
        except Exception as e:
            print(f"❌ วิธีที่ 3 ไม่สำเร็จ: {e}")

# ถ่ายภาพหน้าจอเพื่อตรวจสอบว่ามีการเปลี่ยนแปลงหรือไม่
driver.save_screenshot("after_click.png")

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

# ปิดเบราว์เซอร์
driver.quit()