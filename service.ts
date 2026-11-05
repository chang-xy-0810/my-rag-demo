import express from 'express'
import cors from 'cors'
import * as dotenv from 'dotenv'
import * as fs from 'fs'
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { Document } from "@langchain/core/documents";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { PromptTemplate } from "@langchain/core/prompts";
import { createRetrievalChain } from "@langchain/classic/chains/retrieval";
import { createStuffDocumentsChain } from "@langchain/classic/chains/combine_documents";

dotenv.config();

const app = express();
app.use(express.json());
app.use(cors());

let retrievalChain: any = null;

// 初始化 RAG 检索链（服务启动时加载一次知识库）
async function initRAG() {
  console.log("正在初始化知识库...");
  const text = fs.readFileSync("kb.txt", "utf-8");
  const docs = [new Document({ pageContent: text })];

  const apiKey = process.env.OPENAI_API_KEY;

  const embeddings = new OpenAIEmbeddings({
    apiKey: apiKey,
    model: "gemini-embedding-001",
    configuration: {
      baseURL: "https://generativelanguage.googleapis.com/v1beta/openai/",
    },
  });
  const vectorStore = await MemoryVectorStore.fromDocuments(docs, embeddings);
  const retriever = vectorStore.asRetriever({ k: 1 });

  const model = new ChatOpenAI({
    model: "gemini-3.5-flash-lite",
    temperature: 0,
    apiKey: apiKey,
    configuration: {
      baseURL: "https://generativelanguage.googleapis.com/v1beta/openai/",
    },
  });

  const prompt = PromptTemplate.fromTemplate(`
    你是一个企业内部智能助手。请严格根据下面提供的上下文信息来回答用户的问题。
    如果上下文信息中没有包含答案，请直接回答“抱歉，我的知识库中没有找到相关信息。”。

    上下文信息:
    {context}

    用户问题: {input}
  `);

  const combineDocsChain = await createStuffDocumentsChain({
    llm: model,
    prompt,
    outputParser: new StringOutputParser(),
  });

  retrievalChain = await createRetrievalChain({
    retriever,
    combineDocsChain,
  });
  console.log("知识库初始化完成！");
}

// 接收前端提问的接口
app.post("/api/chat", async (req, res) => {
  try {
    const { message } = req.body;
    if (!message) {
      return res.status(400).json({ error: "消息不能为空" });
    }

    console.log(`收到用户提问: ${message}`);
    const response = await retrievalChain.invoke({ input: message });

    res.json({ answer: response.answer });
  } catch (error: any) {
    console.error("处理出错:", error);
    res.status(500).json({ error: error.message });
  }
});

app.listen(3000, async () => {
  await initRAG();
  console.log("后端服务已启动，监听 http://localhost:3000");
});