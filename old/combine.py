import csv

events_dict = {}
app_events_dict = {}
app_labels_dict = {}
label_categories_dict = {}
phone_brand_dict = {}
device_gender_dict = {}

final_set = []

with open("events.csv") as f:
    dr = csv.DictReader(f, delimiter=',', quotechar='"')
    for line in dr:
        events_dict[line["event_id"]] = line

with open("app_events.csv") as f:
    dr = csv.DictReader(f, delimiter=',', quotechar='"')
    for line in dr:
        app_events_dict[line["event_id"]] = line

with open("app_labels.csv") as f:
    dr = csv.DictReader(f, delimiter=',', quotechar='"')
    for line in dr:
        app_labels_dict[line["app_id"]] = line

with open("label_categories.csv") as f:
    dr = csv.DictReader(f, delimiter=',', quotechar='"')
    for line in dr:
        label_categories_dict[line["label_id"]] = line

with open("phone_brand_device_model.csv") as f:
    dr = csv.DictReader(f, delimiter=',', quotechar='"')
    for line in dr:
        phone_brand_dict[line["device_id"]] = line

with open("gender_age_train.csv") as f:
    dr = csv.DictReader(f, delimiter=',', quotechar='"')
    for line in dr:
        device_gender_dict[line["device_id"]] = line

for event in events_dict:
    """
    event_id, device_id, timestamp, latitude, longitude,
            \
            - phone_brand, device_model,
            - group, gender, age
    \
    - app_id, is_installed, is_active
    \
        - label_id, category
    """

    # device info
    device_id = events_dict[event]["device_id"]

    if device_id in phone_brand_dict:
        device_phone_brand = phone_brand_dict[device_id]["phone_brand"]
        device_model = phone_brand_dict[device_id]["device_model"]
    else:
        device_phone_brand = "XXX"
        device_model = "XXX"

    if device_id in device_gender_dict:
        device_group = device_gender_dict[device_id]["group"]
        device_gender = device_gender_dict[device_id]["gender"]
        device_age = device_gender_dict[device_id]["age"]
    else:
        device_group = "XXX"
        device_gender = "XXX"
        device_age = "XXX"

    # event info (meta)
    event_timestamp = events_dict[event]["timestamp"]
    event_latitude = events_dict[event]["latitude"]
    event_longitude = events_dict[event]["longitude"]

    # event info (app)
    if event in app_events_dict:
        event_app_id = app_events_dict[event]["app_id"]
        event_app_installed = app_events_dict[event]["is_installed"]
        event_app_active = app_events_dict[event]["is_active"]
    else:
        event_app_id = "XXX"
        event_app_installed = "XXX"
        event_app_active = "XXX"

    # app info (label)
    if event_app_id in app_labels_dict:
        app_label_id = app_labels_dict[event_app_id]["label_id"]
    else:
        app_label_id = "XXX"
    # app info (category
    if app_label_id in label_categories_dict:
        app_category = label_categories_dict[app_label_id]["category"]
    else:
        app_category = "XXX"


    event_summary = {
        "device_id": int(device_id),
        "event_id": int(event),
        "device_phone_brand": device_phone_brand,
        "device_model": device_model,
        "event_timestamp": event_timestamp,
        "event_latitude": event_latitude,
        "event_longitude": event_longitude,
        "event_app_id": event_app_id,
        "event_app_installed": event_app_installed,
        "event_app_active": event_app_active,
        "app_label_id": app_label_id,
        "app_category": app_category,
        "device_group": device_group,
        "device_gender": device_gender,
        "device_age": device_age
    }
    final_set.append(event_summary)

def get_id(item):
    return (item["device_id"], item["event_id"])

final_set.sort(key=get_id)

with open('events_combined.csv', 'w') as csvfile:
    fieldnames = [
        "device_id",
        "event_id",
        "device_phone_brand",
        "device_model",
        "event_timestamp",
        "event_latitude",
        "event_longitude",
        "event_app_id",
        "event_app_installed",
        "event_app_active",
        "app_label_id",
        "app_category",
        "device_group",
        "device_gender",
        "device_age"
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for e in final_set:
        writer.writerow(e)
