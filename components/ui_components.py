import streamlit as st
from datetime import datetime, timedelta

def render_train_input_form():
    col1, col2 = st.columns(2)

    with col1:
        train_type = render_train_type_input()
        max_speed = render_max_speed_input()
        num_junction = render_junction_input()
        outbound_return = render_direction_input()
        train_number = render_train_number_input()
        day_of_week = render_day_of_week_input()
        departure_delay = st.number_input("ความล่าช้าจากต้นทาง", min_value=0, max_value=2000, step=1)
        actual_departure_time = st.time_input("เวลาจริงที่ต้นทางออกตัว")
        actual_arrival_time = st.time_input("เวลาจริงที่ถึงปลายทาง")

    with col2:
        departure_time = st.time_input("เวลาออกเดินทางตามกำหนด")

        if "arrival_time" not in st.session_state:
            st.session_state.arrival_time = (datetime.now() + timedelta(minutes=30)).time()

        arrival_time = st.time_input(
            "เวลาถึงจุดหมายตามกำหนด",
            value=st.session_state.arrival_time,
            key="arrival_time"
        )

        railway_line = render_railway_line_input()
        distance = st.number_input("ระยะทาง (กม.)", min_value=31, max_value=1152, value=31)
        num_stations = st.number_input("จำนวนสถานีที่ผ่าน", min_value=2, max_value=93, value=2)
        time_period = render_time_period_input()
        max_delay = st.number_input("ความล่าช้าสูงสุด", min_value=0, max_value=5000, step=1)
        total_delay = st.number_input("รวมเวลาทั้งหมดที่ล่าช้า", min_value=0, max_value=15000, step=1)
        stations_delay = st.number_input("จำนวนสถานีทั้งหมดที่ล่าช้า", min_value=0, max_value=93, step=1)

    # แปลงข้อมูลเวลาเป็นชั่วโมงและนาที
    return {
        # Map to the exact feature names expected by the model
        'Train_type': train_type,
        'Maximum_train_speed': max_speed,
        'Number_of_junctions': num_junction,
        'Outbound_trips_Return_trips': outbound_return,
        'Train_number': train_number,
        'Date': day_of_week,  # This might need to be a full date
        'Departure_delay_origin': departure_delay,
        'Actual_departure_time_origin_hour': actual_departure_time.hour,
        'Actual_departure_time_origin_minute': actual_departure_time.minute,
        'Actual_arrival_time_destination_hour': actual_arrival_time.hour,
        'Actual_arrival_time_destination_minute': actual_arrival_time.minute,
        'Scheduled_departure_time_origin_hour': departure_time.hour,
        'Scheduled_departure_time_origin_minute': departure_time.minute,
        'Scheduled_arrival_time_destination_hour': arrival_time.hour,
        'Scheduled_arrival_time_destination_minute': arrival_time.minute,
        'Railway_line': railway_line,
        'Distance': distance,
        'Number_of_stations': num_stations,
        'Time_period': time_period,
        'Maximum_delay': max_delay,
        'Total_delay_time': total_delay,
        'Number_of_delayed_stations': stations_delay
    }

def render_train_type_input():
    train_types_dict = {
        "รถด่วนพิเศษ": 1, "รถด่วน": 2, "รถเร็ว": 3,
        "รถธรรมดา": 4, "ชานเมือง": 5, "ท้องถิ่น": 6,
        "พิเศษชานเมือง": 7, "พิเศษโดยสาร": 8, "นำเที่ยว": 9,
        "สินค้า": 0, "รถเปล่า": 11, "พิเศษนำเที่ยว": 12,
        "พิเศษ": 13, "พิเศษช่วยอันตราย": 14,
    }
    train_type_display = st.selectbox("ประเภทรถไฟ", list(train_types_dict.keys()))
    return train_types_dict[train_type_display]

def render_max_speed_input():
    max_speed_types_dict = {
        "60": 60, "100": 100, "120": 120, "160": 160,
    }
    max_speed_display = st.selectbox("ความเร็วสูงสุด (กม./ชม.)", list(max_speed_types_dict.keys()))
    return max_speed_types_dict[max_speed_display]

def render_junction_input():
    num_junction_types_dict = {
          "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
    }
    num_junction_display = st.selectbox("จำนวนทางแยก", list(num_junction_types_dict.keys()))
    return num_junction_types_dict[num_junction_display]

def render_direction_input():
    outbound_return_types_dict = {
        "ขาไป": 1, "ขากลับ": 2,
    }
    outbound_return_display = st.selectbox("ทางออก-กลับ", list(outbound_return_types_dict.keys()))
    return outbound_return_types_dict[outbound_return_display]

def render_train_number_input():
    train_number_types_dict = {
    '7': 7, '8': 8, '9': 9, '10': 10, '13': 13, '14': 14, '21': 21, '22': 22,
    '23': 23, '25': 25, '26': 26, '31': 31, '32': 32, '37': 37, '38': 38,
    '39': 39, '40': 40, '43': 43, '45': 45, '46': 46, '51': 51, '71': 71,
    '75': 75, '76': 76, '83': 83, '84': 84, '85': 85,
    '86': 86, '102': 102, '107': 107, '109': 109, '111': 111,
    '112': 112, '133': 133, '134': 134, '135': 135, '136': 136,
    '139': 139, '140': 140, '141': 141, '142': 142, '147': 147,
    '148': 148, '167': 167, '168': 168, '169': 169, '170': 170,
    '171': 171, '172': 172, '201': 201, '202': 202, '207': 207,
    '208': 208, '209': 209, '210': 210, '211': 211, '212': 212,
    '233': 233, '234': 234, '251': 251, '252': 252, '254': 254,
    '255': 255, '257': 257, '258': 258, '259': 259, '260': 260,
    '261': 261, '262': 262, '275': 275, '276': 276, '277': 277,
    '278': 278, '279': 279, '280': 280, '281': 281, '282': 282,
    '283': 283, '284': 284, '301': 301, '302': 302, '303': 303,
    '304': 304, '313': 313, '314': 314, '317': 317, '318': 318,
    '339': 339, '340': 340, '341': 341, '342': 342, '351': 351,
    '352': 352, '355': 355, '356': 356, '367': 367, '368': 368,
    '371': 371, '372': 372, '379': 379, '380': 380, '383': 383,
    '384': 384, '388': 388, '389': 389, '390': 390, '391': 391,
    '401': 401, '402': 402, '403': 403, '405': 405, '406': 406,
    '407': 407, '408': 408, '410': 410, '415': 415, '416': 416,
    '417': 417, '418': 418, '419': 419, '420': 420, '421': 421,
    '422': 422, '423': 423, '424': 424, '425': 425, '426': 426,
    '427': 427, '428': 428, '429': 429, '430': 430, '431': 431,
    '432': 432, '433': 433, '434': 434, '439': 439, '440': 440,
    '445': 445, '446': 446, '447': 447, '448': 448, '451': 451,
    '452': 452, '453': 453, '454': 454, '455': 455, '456': 456,
    '463': 463, '464': 464, '485': 485, '486': 486, '489': 489,
    '490': 490, '721': 721, '722': 722, '831': 831, '832': 832,
    '833': 833, '834': 834, '835': 835, '836': 836, '837': 837,
    '838': 838, '839': 839, '840': 840, '841': 841, '842': 842,
    '843': 843, '844': 844, '845': 845, '846': 846, '847': 847,
    '848': 848, '849': 849, '850': 850, '851': 851, '852': 852,
    '853': 853, '854': 854, '855': 855, '856': 856, '857': 857,
    '858': 858, '859': 859, '860': 860, '909': 909, '910': 910,
    '911': 911, '912': 912, '947': 947, '948': 948, '949': 949,
    '950': 950, '985': 985, '986': 986, '997': 997, '998': 998,
    '1115': 1115, '1116': 1116, '1117': 1117, '1118': 1118, '1123': 1123,
    '1124': 1124, '1129': 1129, '1132': 1132, '4302': 4302, '4303': 4303,
    '4304': 4304, '4305': 4305, '4306': 4306, '4307': 4307, '4308': 4308,
    '4309': 4309, '4310': 4310, '4311': 4311, '4312': 4312, '4313': 4313,
    '4314': 4314, '4315': 4315, '4316': 4316, '4317': 4317, '4320': 4320,
    '4321': 4321, '4322': 4322, '4323': 4323, '4324': 4324, '4325': 4325,
    '4326': 4326, '4327': 4327, '4328': 4328, '4329': 4329, '4340': 4340,
    '4341': 4341, '4342': 4342, '4343': 4343, '4344': 4344, '4345': 4345,
    '4346': 4346, '4347': 4347, '4380': 4380, '4381': 4381, '4382': 4382,
    '4383': 4383, '4384': 4384, '4385': 4385, '4386': 4386, '4387': 4387
    }
    train_number_display = st.selectbox("หมายเลขรถไฟ", list(train_number_types_dict.keys()))
    return train_number_types_dict[train_number_display]

def render_day_of_week_input():
    day_of_week_types_dict = {
        "จันทร์": 1, "อังคาร": 2, "พุธ": 3, "พฤหัสบดี": 4, "ศุกร์": 5, "เสาร์": 6, "อาทิตย์": 7,
    }
    day_of_week_display = st.selectbox("วันในสัปดาห์", list(day_of_week_types_dict.keys()))
    return day_of_week_types_dict[day_of_week_display]

def render_railway_line_input():
    railway_line_types_dict = {
        "สายเหนือ": 1,"สายตะวันออกเฉียงเหนือ": 2,"สายตะวันออก": 3,
        "สายใต้": 4,"วงเวียนใหญ่ - มหาชัย": 5,"บ้านแหลม - แม่กลอง": 6,
        "เชื่อมต่อสายสีแดง": 7,"สายใต้ - สินค้า": 8,"ข้ามภูมิภาค": 9
    }
    railway_line_display = st.selectbox("สายรถไฟ", list(railway_line_types_dict.keys()))
    return railway_line_types_dict[railway_line_display]

def render_time_period_input():
    time_period_types_dict = {
      "กลางวัน": 1,"กลางคืน": 2,
    }
    time_period_display = st.selectbox("ช่วงเวลา", list(time_period_types_dict.keys()))
    return time_period_types_dict[time_period_display]
