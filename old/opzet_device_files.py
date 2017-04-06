import pandas as pd

unique_devices = np.unique(data['device_id'])

for u in unique_devices[:5]:
    device_data = data[data['device_id']==u]
    
    data_csv = open('./device_data_files/'+unicode(u)+'.txt', 'w')
    device_data.to_csv(data_csv, sep = ' ', header=False, index=False, columns=['device_phone_brand', 'device_model'], encoding = 'utf-8')
    data_csv.close()