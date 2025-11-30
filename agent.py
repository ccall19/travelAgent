import os

from langchain.agents import create_agent, AgentState
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver  
from langchain.agents.middleware import before_model
from langgraph.runtime import Runtime
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from pydantic import BaseModel, Field
import pandas as pd
import requests
import fastapi
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import yaml
import asyncio
from typing import Any

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

model = ChatOpenAI(
    model = "glm-4.6",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    openai_api_key=config["openai_api_key"],
    extra_body={
        "thinking": {"type": "disabled"} 
    }
)

class CityAdcode(BaseModel):
    adcode: str = Field(description="行政区划代码")
    type: str = Field(description="查询类型：'base'代表实时天气（现在），'all'代表天气预报（未来3天）", default='base')

class InputCityDecode(BaseModel):
    province: str = Field(description="省份名称")
    city: str = Field(description="城市名称")
    district: str = Field(description="区县名称")

data = pd.read_excel('AMap_adcode_citycode/AMap_adcode_citycode.xlsx')

@tool(args_schema=InputCityDecode)
def query_adcode(province: str, city: str, district: str) -> str:
    """Query the adcode for a given province, city, and district."""
    tar = 0
    if province:
        for i in range(len(data)):
            if data['中文名'][i] == province:
                adcode = data['adcode'][i] 
                tar = i
                break
    if city:
        for i in range(tar, len(data)):
            if data['中文名'][i] == city:
                adcode = data['adcode'][i] 
                tar = i
                break
            if data['中文名'][i].endswith('省'):
                break 
    if district:
        for i in range(tar, len(data)):
            if data['中文名'][i] == district:
                adcode = data['adcode'][i] 
                tar = i
                break
            if data['中文名'][i].endswith('市'):
                break
    if 'adcode' not in locals():
        return ""
    return str(adcode)

@tool(args_schema=CityAdcode) 
def get_weather(adcode: str, type: str = 'base') -> str:
    """Get the weather information for a given city."""
    if len(adcode) == 0:
        return f"City with adcode {adcode} not found."

    key = config["gaode_api_key"]
    url = f"https://restapi.amap.com/v3/weather/weatherInfo?city={adcode}&key={key}&extensions={type}&output=JSON"

    response = requests.get(url)
    weather_data = response.json()
    return weather_data

@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Keep only the last few messages to fit context window."""
    messages = state["messages"]
    if len(messages) <= 9:
        return None  # No changes needed
    
    first_msg = messages[0]
    recent_messages = messages[-9:] if len(messages) % 2 == 0 else messages[-10:]
    new_messages = [first_msg] + recent_messages

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }

agent_weather = create_agent(
    model = model,
    tools = [query_adcode, get_weather],
    system_prompt="""
    你是一个天气查询代理，负责根据用户的问题获取指定城市的天气信息。

    步骤：
    1. 分析用户查询中的地名，将其标准化为三级行政单位：省、市、区县。
    2. 使用“查询adcode”工具获取adcode。
    3. 判断用户意图：
       - 如果用户问“现在”、“当前”的天气，调用“获取天气”工具时，type参数传 'base'。
       - 如果用户问“明天”、“后天”、“未来几天”的天气，调用“获取天气”工具时，type参数传 'all'。
    4. 使用“获取天气”工具查询。
    5. 将天气信息以清晰、友好的方式反馈给用户。如果是预报，请列出具体的日期和天气情况。
    
    注意：地名必须是标准的三级名称；如果用户只提供部分信息，你需要推断完整的三级结构。始终使用工具，不要直接回答。
    """
)

@tool(
    "call_weather_agent",
    description="调用天气代理以获取天气信息。",
)
def call_weather_agent(query: str) -> str:
    """Helper function to call the weather agent."""
    response = agent_weather.invoke(
        {
            "messages": [{"role": "user", "content": query}]
        }
    )
    return response['messages'][-1].content

class POIRequest(BaseModel):
    keywords: str

@tool(args_schema=POIRequest)
def get_poi(keywords: str) -> str:
    """Get point of interest (POI) information for Tiananmen, Beijing."""
    key = os.getenv("GAODE_API_KEY")
    url = f"https://restapi.amap.com/v5/place/text?key={key}&keywords={keywords}&show_fields=business"
    response = requests.get(url)
    poi_data = response.json()
    return poi_data

agent_travel = create_agent(
    model = model,
    tools = [get_poi],
    system_prompt="""
    你是一个旅游信息查询代理，负责根据用户的问题获取指定地点的POI信息。
    """
)

@tool(
    "call_travel_agent",
    description="调用旅游代理以获取POI信息。",
)
def call_travel_agent(query: str) -> str:
    """Helper function to call the travel agent."""
    response = agent_travel.invoke(
        {
            "messages": [{"role": "user", "content": query}]
        }
    )
    return response['messages'][-1].content

agent_supervisor = create_agent(
    model = model,
    tools = [call_weather_agent, call_travel_agent],
    middleware=[trim_messages],
    checkpointer=InMemorySaver(), 
    system_prompt="You are a personal assistant. Use the right tools for each task."
)

app = fastapi.FastAPI()

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"], 
)

def to_sse_chunk(text: str) -> str:
    return "data: " + text.replace("\n", "\ndata: ") + "\n\n"

@app.get("/query-weather")
async def query_weather(userid: str, channel_id: str, query: str):
    """API endpoint to query weather information using SSE."""
    if not query:
        return fastapi.responses.JSONResponse(
            {"error": "Query parameter is required"},
            status_code=400
        )
    async def event_generator(user_query: str):
        """Generator function to stream response tokens with SSE format."""
        try:
            for token, metadata in agent_supervisor.stream(  
                {"messages": [{"role": "user", "content": user_query}]},
                {"configurable": {"thread_id": userid + channel_id}},
                stream_mode="messages",
            ):
                if metadata['langgraph_node'] == 'model':
                    if hasattr(token, 'content_blocks') and len(token.content_blocks) >= 1 and token.content_blocks[0]['type'] == 'text':
                        text = token.content_blocks[0]['text']
                        yield to_sse_chunk(text)  # SSE 格式
                    elif isinstance(token.content, str):
                        yield to_sse_chunk(token.content)
        except Exception as e:
            yield f"data: 错误: {str(e)}\n\n"
    return StreamingResponse(event_generator(query),media_type="text/event-stream") # SSE 格式

async def generate_data():
    for i in range(10):
        yield f"Chunk {i}\n"  # 每次返回一小段数据
        await asyncio.sleep(1)  # 模拟延迟

# 创建流式接口
@app.get("/stream")
async def stream_data():
    return StreamingResponse(generate_data(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
