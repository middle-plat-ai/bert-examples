# -*- coding: utf-8 -*-

import json

f = open('output.json', encoding='utf8')
output = json.load(f)

print(output.keys())
print(len(output["features"]))
for value in output["features"]:
    print(len(value["layers"]))
    print(len(value["layers"][2]["values"]))
    print(value["layers"][2]["values"])