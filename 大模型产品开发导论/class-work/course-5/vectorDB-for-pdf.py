import os
from dotenv import load_dotenv

# 从.env文件中 加载环境变量
load_dotenv()

# 加载大模型（聊天模型）
from langchain_community.chat_models import ChatZhipuAI
llm = ChatZhipuAI(
    temperature=0.3,
    model="glm-4-plus",
    zhipuai_api_key=os.getenv("ZHIPUAI_API_KEY"),
)

# RAG（Retrieval检索 + Generation生成）
# 数据准备阶段：数据提取（文档加载） 》〉 文本分割 》〉 向量化 》〉 数据入库（向量数据库存储 采用本地chroma数据库存储）
# pdf文档读入
from langchain_community.document_loaders import PyPDFLoader

file_path = "class-work/course-5/data/2024年金融科技生态蓝皮书.pdf"
loader_pdf = PyPDFLoader(file_path)
docs_pdf = loader_pdf.load()

# pdf文本分隔
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter_pdf = RecursiveCharacterTextSplitter(
    chunk_size = 1000, 
    chunk_overlap = 200,
    length_function = len,
    separators= ["\n\n", "\n", " ", "."]
)

docs_pdf = text_splitter_pdf.split_documents(docs_pdf)

# print(docs_pdf[2]) #打印分割后的文本
# sum([len(doc.page_content) for doc in docs_pdf]) 计算字数

# 文档词向量化 与 向量数据库存储
#加载zhipuai embedding模型
from langchain_community.embeddings import ZhipuAIEmbeddings

embeddings_model =ZhipuAIEmbeddings(
    model='embedding-3',
    api_key=os.getenv("ZHIPUAI_API_KEY"),
)

# 因为emded_documents方法 调用的字符串列表 所以texts是将docs_pdf转化为字符串列表
# texts = [doc.page_content for doc in docs_pdf]
# # 向量化 其结果在内存中，临时性的
# embeddings = embeddings_model.embed_documents(texts)

# print(embeddings[0]) #打印向量化后的数据
# print(len(embeddings)) #打印向量化后的数据长度
# print(len(embeddings[0])) #打印向量化后的数据

# 向量数据库存储
from langchain.vectorstores import Chroma

# 该指令是创建向量数据库 这里的persist_directory是指向量数据库存储的路径
# vectordb = Chroma.from_documents( 
#     documents=docs_pdf,
#     embedding=embeddings_model,
#     persist_directory="./db/chroma_db",
# )

# 已有向量数据库 导入
vectordb_load = Chroma( 
    persist_directory="class-work/course-5/db/chroma_db",
    embedding_function=embeddings_model,
)

# 通过 已有向量数据库 检索相关文档
res = vectordb_load.similarity_search("金融科技是什么？")
print(res[0].page_content) #打印检索到的相关文档
