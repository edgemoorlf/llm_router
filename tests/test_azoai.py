import requests

# 配置参数
API_KEY = "2PZVEgzOujTs9aT2lxYfq2yuunnbUmjud0HzPbZAXzNFLWgdf0WHJQQJ99BCACHYHv6XJ3w3AAABACOGiMb8"  # 替换为你的API密钥
API_BASE = "https://eastus2-0317-27-1-1120.openai.azure.com/"
DEPLOYMENT_NAME = "gpt-4o"
DEPLOYMENT_ENDPOINT = f"{API_BASE}/openai/deployments/{DEPLOYMENT_NAME}/chat/completions?api-version=2024-08-01-preview"
PROXY_URL = "http://fd955bb413ae99:a3af88e6@15.204.52.62:5001"



API_VERSION = "2024-08-01-preview"  # 使用最新API版本

headers = {
    "Content-Type": "application/json",
    "api-key": API_KEY
}

payload = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant with expertise."},
        {"role": "user", "content": "小泽玛丽亚是谁？"}
    ],
    "temperature": 0.7,
    "max_tokens": 800,
}

# 发送请求
response = requests.post(
    f"{DEPLOYMENT_ENDPOINT}",
    headers=headers,
    json=payload,
    proxies={
        "http": PROXY_URL    
    }
)

# 处理响应
if response.status_code == 200:
    reply = response.json()['choices'][0]['message']['content']
    print("Assistant:", reply)
else:
    print(f"Error {response.status_code}: {response.text}")
