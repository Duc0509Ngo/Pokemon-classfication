import requests

resp = requests.post("http://localhost:5000/predict",
                     files={"file": open('../10.png','rb')})


print(resp.json())