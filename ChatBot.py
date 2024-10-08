from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from config import GLM_API_KEY,TAVILY_API_KEY
import  os

# 定义状态类型，继承自 TypedDict，并使用 add_messages 函数将消息追加到现有列表
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 创建一个状态图对象，传入状态定义
graph_builder = StateGraph(State)

from langchain_openai import ChatOpenAI

# 初始化一个 GPT-4o-mini 模型
#chat_model = ChatOpenAI(model="gpt-4o-mini")
chat_model  = ChatOpenAI(
        temperature=0.95,
        model="glm-4-flash",
        api_key=GLM_API_KEY,
        base_url="https://open.bigmodel.cn/api/paas/v4/")


# 定义聊天机器人的节点函数，接收当前状态并返回更新的消息列表
def chatbot(state: State):
    return {"messages": [chat_model.invoke(state["messages"])]}

# 第一个参数是唯一的节点名称，第二个参数是每次节点被调用时的函数或对象
graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# 编译状态图并生成可执行图对象
graph = graph_builder.compile()

# 你可以使用 get_graph 方法来可视化图，并结合 draw 方法（如 draw_ascii 或 draw_png）
from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    pass

# 开始一个简单的聊天循环
# while True:
#     # 获取用户输入
#     user_input = input("User: ")
#
#     # 可以随时通过输入 "quit"、"exit" 或 "q" 退出聊天循环
#     if user_input.lower() in ["quit", "exit", "q"]:
#         print("Goodbye!")  # 打印告别信息
#         break  # 结束循环，退出聊天
#
#     # 将每次用户输入的内容传递给 graph.stream，用于聊天机器人状态处理
#     # "messages": ("user", user_input) 表示传递的消息是用户输入的内容
#     for event in graph.stream({"messages": ("user", user_input)}):
#
#         # 遍历每个事件的值
#         for value in event.values():
#             # 打印输出 chatbot 生成的最新消息
#             print("Assistant:", value["messages"][-1].content)


from langchain_community.tools.tavily_search import TavilySearchResults


os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
# 定义 Tavily 搜索工具，最大搜索结果数设置为 2
tool = TavilySearchResults(max_results=2)
tools = [tool]

# 测试工具调用
print(tool.invoke("最新的AI新闻有哪些?"))