import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import * as fs from "fs";
import dotenv from "dotenv";
import { createRequire } from "module";

dotenv.config();

// pdf-parse is CommonJS, so we load it this way
const require = createRequire(import.meta.url);
const pdf = require("pdf-parse");

/**
 * Analyze a resume PDF using Retrieval-Augmented Generation (RAG)
 * @param {string} filePath - Path to uploaded PDF
 * @returns {{ praising: string, critical: string }}
 */
export async function analyzeResume(filePath) {
  try {
    // 1. Read and parse PDF
    const buffer = fs.readFileSync(filePath);
    const pdfData = await pdf(buffer);
    const resumeText = pdfData.text;

    if (!resumeText || resumeText.trim().length === 0) {
      throw new Error("No readable text found in PDF.");
    }

    // 2. Split resume into chunks (IMPROVED)
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

    // 4. LLM (low temperature for factual accuracy)
    const llm = new ChatOpenAI({
      model: "gpt-4o",
      temperature: 0.2,
    });

    // 5. Prompt templates
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

    // 6. RAG chain builder (NO deprecated imports)
    const buildChain = (prompt) =>
      RunnableSequence.from([
        async () => {
          const retrievedDocs =
            await retriever.getRelevantDocuments("Analyze the resume");
          return {
            context: retrievedDocs
              .map((doc) => doc.pageContent)
              .join("\n\n"),
          };
        },
        prompt,
        llm,
      ]);

    // 7. Run both analyses
    const praisingResult = await buildChain(praisingPrompt).invoke({});
    const criticalResult = await buildChain(criticalPrompt).invoke({});

    // 8. Return results to backend / frontend
    return {
      praising: praisingResult.content,
      critical: criticalResult.content,
    };
  } catch (error) {
    console.error("RAG ERROR:", error.message);
    throw error;
  }
}
