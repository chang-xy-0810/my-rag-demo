import { OpenAIEmbeddings, ChatOpenAI} from "@langchain/openai";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { Document } from "@langchain/core/documents";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { PromptTemplate } from "@langchain/core/prompts";
import { createRetrievalChain } from "@langchain/classic/chains/retrieval";
import { createStuffDocumentsChain } from "@langchain/classic/chains/combine_documents";
import * as fs from "fs";
import * as dotenv from "dotenv";

// 读取环境变量
dotenv.config();

async function main(): Promise<void> {
  // 1. 类型安全检查环境变量
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    console.error("错误: 请先在 .env 文件或环境变量中设置 OPENAI_API_KEY！");
    return;
  }

  console.log("正在加载私有知识库文件...");
  // 2. 读取文本文件
  const text: string = fs.readFileSync("kb.txt", "utf-8");

  // 3. 构建文档对象数组
  const docs: Document[] = [new Document({ pageContent: text })];

  console.log("正在进行向量化 (Embedding) 并存入内存向量数据库...");
  // 4. 初始化向量数据库与 Embedding 模型
  const embeddings = new OpenAIEmbeddings({
    apiKey: apiKey,
    model: "gemini-embedding-001",
    configuration: {
      baseURL: "https://generativelanguage.googleapis.com/v1beta/openai/",
    },
  });
  const vectorStore = await MemoryVectorStore.fromDocuments(docs, embeddings);

  // 5. 创建检索器 (Retriever)
  const retriever = vectorStore.asRetriever({
    k: 1, // 仅检索最相关的 1 个切片
  });

  console.log("初始化大模型...");
  // 6. 初始化大模型 (Gemini OpenAI 兼容接口)
  const model = new ChatOpenAI({
    model: "gemini-3.5-flash-lite",
    temperature: 0,
    apiKey: apiKey,
    configuration: {
      baseURL: "https://generativelanguage.googleapis.com/v1beta/openai/",
    },
  });

  // 7. 编写 RAG 提示词模板
  const prompt = PromptTemplate.fromTemplate(`
    你是一个企业内部智能助手。请严格根据下面提供的上下文信息来回答用户的问题。
    如果上下文信息中没有包含答案，请直接回答“抱歉，我的知识库中没有找到相关信息。”，绝对不要自己瞎编。

    上下文信息:
    {context}

    用户问题: {input}
  `);

  // 8. 组装 RAG 检索链
  const combineDocsChain = await createStuffDocumentsChain({
    llm: model,
    prompt,
    outputParser: new StringOutputParser(),
  });

  const retrievalChain = await createRetrievalChain({
    retriever,
    combineDocsChain,
  });

  // ==================== 9. 开始提问测试 ====================
  const question = "Mr.chang有多少年开发经验？公司年假怎么规定的？";
  console.log(`\n用户提问: "${question}"`);

  console.log("正在检索知识库并生成回答...\n");
  
  // 强类型调用
  const response = await retrievalChain.invoke({
    input: question,
  });

  console.log("========== AI 的回答 ==========");
  console.log(response.answer);
  console.log("===============================");
}

main().catch((error: unknown) => {
  console.error("运行出错:", error);
});