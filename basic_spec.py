from genericpath import exists
import pickle as pk

if exists("basic_setting.spec"):
    with open("basic_setting.spec", "rb") as f:
        setting = pk.load(f)
        print(setting)
    
print("BLUR_BIAS, IMG_RESIZE, IMG_SHEER,DIGITS_HEIGHT_MAX,DIGITS_HEIGHT_MIN,MISSING_FAULT_LIMIT_C, MISSING_FAULT_LIMIT_E,LOGGING_TIME_DIST,LOGGING_TIME_CUT_LOWER,LOGGING_TIME_CUT_UPPER. take apart values with comma ','.")
values = input(":: ").split(",")[:10]

values = list(map(float, values))


indexes_int = [0,1]
for index in indexes_int:
    values[index] = int(values[index])

with open("basic_setting.spec", "wb") as f:
    setting = values
    pk.dump(setting, f)
    