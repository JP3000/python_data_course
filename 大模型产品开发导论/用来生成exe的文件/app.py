import streamlit as st
import os
from langchain_community.llms import Tongyi
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory



st.title("📖 langchain——通义千问")

if 'history' not in st.session_state:
    st.session_state['history'] = ConversationBufferMemory()

temp = st.sidebar.select_slider(
    "选择温度",
    options=[i/10 for i in range(0,10)],value=0)
st.write("温度为", temp)

if tongyi := st.sidebar.text_input("通义千问"):
    model_ty = Tongyi(temperature=temp,dashscope_api_key=tongyi)

    conversation = ConversationChain(
        llm=model_ty,  
        memory=st.session_state['history']
    )
    st.write("模型已经就绪请输入问题")
else:
    st.write("请在左侧输入key")

for i in st.session_state['history'].chat_memory.messages:
    st.chat_message(i.type).write(i.content)

if p_y :=st.chat_input():
    st.chat_message("user").write(p_y)
    res=conversation.invoke(p_y)
    st.chat_message("AI").write(res['response'])
