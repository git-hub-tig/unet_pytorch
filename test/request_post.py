import requests
import time
import datetime

start = time.time()
resp = requests.post(
    "http://localhost:5000/predict",
    files={
        "file": open("/Users/wqs/Downloads/interview-mleng/images/image6.jpg", "rb")
    },
)
end = time.time()
print("\nInference time took {}".format(datetime.timedelta(seconds=end - start)))
