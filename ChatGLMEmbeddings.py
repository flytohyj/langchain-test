from langchain.embeddings import ZhipuAIEmbeddings
from transformers import AutoTokenizer, AutoModel
import torch


class ChatGLMEmbeddings(ZhipuAIEmbeddings):
    def __init__(self, model_name="D:\\module\\BAAI-bge-large-zh-v1.5"):
        """
        初始化 ChatGLMEmbeddings 类.

        :param model_name: 预训练模型的路径或名称 (str)
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

    def embed(self, texts):
        """
        生成文本的嵌入.

        :param texts: 输入的文本列表 (list of str)
        :return: 文本的嵌入数组 (numpy.ndarray)
        """
        # 确保输入是列表
        if not isinstance(texts, list):
            raise ValueError("Input must be a list of strings.")

        # 编码输入文本
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

        # 关闭梯度计算，以节省内存
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 获取最后一层隐藏状态并计算平均作为嵌入
        embeddings = outputs.last_hidden_state.mean(dim=1)  # 根据需要选择不同的方式获取嵌入
        return embeddings.numpy()  # 转换为 NumPy 数组


# 用法示例
if __name__ == "__main__":
    embedding_model = ChatGLMEmbeddings(model_name="D:\\module\\BAAI-bge-large-zh-v1.5")

    # 示例文本
    example_texts = ["深度学习是一种机器学习技术", "自然语言处理是人工智能的重要领域"]

    # 生成嵌入
    embeddings = embedding_model.embed(example_texts)
    print(embeddings)
