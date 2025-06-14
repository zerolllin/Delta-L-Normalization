import requests

# 服务地址
url = "http://127.0.0.1:6284/is_equal"

# 构造要对比的两个 LaTeX 字符串示例
data = {
    "str1": "\\frac{3}{5}",
    "str2": "0.6"
}

response = requests.post(url, json=data)

if response.status_code == 200:
    result = response.json()
    print("Equality:", result["equal"])
else:
    print("Error:", response.text)