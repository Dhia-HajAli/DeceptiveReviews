import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'review':'My family and I are huge fans of this place. The staff is super nice, and the food is great. The chicken is very good, and the garlic sauce is perfect. Ice cream topped with fruit is delicious too. Highly recommended!'})

print(r.json())