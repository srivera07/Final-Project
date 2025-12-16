import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";
import { formatDocumentsAsString } from "langchain/util/document";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import * as fs from "fs";
import dotenv from "dotenv";
import { createRequire } from "module";

dotenv.config();

// Required for pdf-parse (CommonJS module)
const require = createRequire(import.meta.url);
const pdf = require("pdf-parse");

/**
 * Analyze a resume PDF using Retrieval-Augmented Generation (RAG)
 * @param {string} filePath - Path to uploaded PDF
 * @returns {Object} praising and critical analysis
 */
export async function analyzeResume(filePath) {
  try {
    // 1. Read PDF
    const buffer = fs.readFileSync(filePath);
    const pdfData = await pdf(buffer);
    const resumeText = pdfData.text;

    if (!resumeText || resumeText.trim().length === 0) {
      throw new Error("No readable text found in PDF.");
    }

    // 2. Split text into chunks (IMPROVED)
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 800,
      chunkOverlap: 200,
    });

    const docs = await splitter.createDocuments([resumeText]);

    // 3. Create embeddings and vector store
    const embeddings = new OpenAIEmbeddings();
    const vectorStore = await MemoryVectorStore.fromDocuments(
      docs,
      embeddings
    );

    const retriever = vectorStore.asRetriever({ k: 4 });

    // 4. LLM (lower temperature = more factual)
    const llm = new ChatOpenAI({
      model: "gpt-4o",
      temperature: 0.2,
    });

    // 5. Prompts
    const praisingPrompt = PromptTemplate.fromTemplate(`
You are an enthusiastic hiring manager.
Using ONLY the resume context below, write a concise and compelling summary
highlighting the candidate’s strengths, skills, and achievements.

Context:
{context}

Answer:
`);

    const criticalPrompt = PromptTemplate.fromTemplate(`
You are a cautious and objective technical recruiter.
Using ONLY the resume context below, identify weaknesses, gaps,
or unclear areas. If none exist, say the resume appears strong.

Context:
{context}

Answer:
`);

    // 6. RAG chain builder
    const buildChain = (prompt) =>
      RunnableSequence.from([
        {
          context: RunnableSequence.from([
            () => "Analyze the resume",
            retriever,
            formatDocumentsAsString,
          ]),
        },
        prompt,
        llm,
      ]);

    // 7. Run chains
    const praisingResult = await buildChain(praisingPrompt).invoke({});
    const criticalResult = await buildChain(criticalPrompt).invoke({});

    // 8. Return results to frontend
    return {
      praising: praisingResult.content,
      critical: criticalResult.content,
    };

  } catch (error) {
    console.error("RAG ERROR:", error.message);
    throw error;
  }
}
