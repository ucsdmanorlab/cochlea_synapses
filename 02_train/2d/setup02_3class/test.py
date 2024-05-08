import neuroglancer
import numpy as np
import os
import webbrowser

neuroglancer.set_static_content_source() # default is to use Google Client
viewer = neuroglancer.Viewer()

with viewer.txn() as s:
    s.layers["Allen Mouse Brain Atlas"] = neuroglancer.SegmentationLayer(
        source='precomputed://gs://wanglab-pma/allenatlas_2017',
    )

url = str(viewer)
print(url)
if os.environ.get("DISPLAY"):
    try:
        webbrowser.open_new_tab(url)
    except:
        webbrowser.open_new(url)

print("Press ENTER to quit")

