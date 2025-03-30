
    # "Authorization": "sk-2MSaRq9inuesh7xv9in0dzlz8BFWae4FfRc7urbWibjhDYRc",
# url = 'http://47.89.180.170:5000/v1'
    # "model": "o3-mini",  # "gpt-4",

models = ["gpt-4o", "gpt-4o-2024-08-06", "gpt-4o-2024-11-20"]

m1 = [
        {
            'role': 'system', 
            'content': """\n===== RULES OF USER =====\nNever forget you are a user and I am a assistant. Never flip roles! You will always instruct me. We share a common interest in collaborating to successfully complete a task.\nI must help 
you to complete a difficult task.\nYou must instruct me based on my expertise and your needs to solve the task step by step. The format of your instruction is: `Instruction: [YOUR INSTRUCTION]`, where "Instruction" describes a sub-task or question.\nYou must give me one instruction at a time.\nI must write a response that appropriately solves the requested ins
truction.\nYou should instruct me not ask me questions.\n\nPlease note that the task may be very complicated. Do not attempt to solve the task by single step. You must instruct me to find the answer step by step.\nHere are some tips that will help you to give more valuable instructions about our task to me:\n<tips>\n- I have various tools to use, such as searc
h toolkit, web browser simulation toolkit, document relevant toolkit, code execution toolkit, etc. Thus, You must think how human will solve the task step-by-step, and give me instructions just like that. For example, one may first use google search to get some initial information and the target url, then retrieve the content of the url, or do some web browser
 interaction to find the answer.\n- Although the task is complex, the answer does exist. If you can’t find the answer using the current scheme, try to re-plan and use other ways to find the answer, e.g. using other tools or methods that can achieve similar results.\n- Always remind me to verify my final answer about the overall task. This work can be done by u
sing multiple tools(e.g., screenshots, webpage analysis, etc.), or something else.\n- If I have written code, please remind me to run the code and get the result.\n- Search results typically do not provide precise answers. It is not likely to find the answer directly using search toolkit only, the search query should be concise and focuses on finding sources r
ather than direct answers, as it always need to use other tools to further process the url, e.g. interact with the webpage, extract webpage content, etc. \n- If the question mentions youtube video, in most cases you have to process the content of the mentioned video.\n- For downloading files, you can either use the web browser simulation toolkit or write codes
 (for example, the github content can be downloaded via https://raw.githubusercontent.com/...).\n- Flexibly write codes to solve some problems, such as excel relevant tasks.\n</tips>\n\nNow, here is the overall task: <task>If we assume all articles published by Nature in 2020 (articles, only, not book reviews/columns, etc) relied on statistical significance to
 justify their findings and they on average came to a p-value of 0.04, how many papers would be incorrect as to their claims of statistical significance? Round the value up to the next integer.</task>. Never forget our task!\n\nNow you must start to instruct me to solve the task step-by-step. Do not add anything else other than your instruction!\nKeep giving m
e instructions until you think the task is completed.\nWhen the task is completed, you must only reply with a single word <TASK_DONE>.\nNever say <TASK_DONE> unless my responses have solved your task.\n """
        }, 
        {
            'role': 'user', 
            'content': """\nNow please give me instructions to solve over overall task step by step. If the task requires some specific knowledge
, please instruct me to use tools to complete the task.\n"""
        }
    ]


m2 = [
        {
            "role": "user",
            "content": "简短分析下国足赢得世界杯的可能性的百分比，具体到数字"
        }
    ]
messages = [

m1,
m2

]

# 45.78.211.2
import requests

headers = {
    "Content-Type": "application/json"
}
url = 'http://localhost:3010/v1/chat/completions'
data = {
    "model": "gpt-4o-2024-08-06",
    "messages": [
        {
            "role": "user",
            "content": "简短分析下国足赢得世界杯的可能性的百分比，具体到数字"
        }
    ],
    "max_tokens": 1000,
    "stream": False,
    "logprobs": True
}
response = requests.post(url, headers=headers, json=data)
print(response.json())
reply = response.json()['choices'][0]['message']['content']
print("Assistant:", reply)
