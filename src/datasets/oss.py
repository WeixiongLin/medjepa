from io import BytesIO
from petrel_client.client import Client
from nibabel import FileHolder, Nifti1Image
from gzip import GzipFile
import numpy as np



def open_npy(url):
    conf_path = '~/petreloss.conf'
    client = Client(conf_path) # client搭建了和ceph通信的通道
    data = client.get(url)
    value_buf = memoryview(data)
    iostr = BytesIO(value_buf)
    img_array = np.load(iostr)
    return img_array

