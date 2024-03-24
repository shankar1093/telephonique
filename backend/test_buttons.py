import json, requests

def alexa_switch(x):
  y = 2
  if x == "On":
    y = 1
  body = json.dumps({
    "virtualButton": y,
    "accessCode": "vbac.69P12388647CLQLRRDI"
  })

  requests.post(url = "https://api.virtualbuttons.com/v1", data = body)

alexa_switch("Off")