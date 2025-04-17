import { TextLoader } from "langchain/document_loaders/fs/text";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import fs from "fs/promises";
import { configDotenv } from "dotenv";
import { ChatOllama, OllamaEmbeddings } from "@langchain/ollama";
import { QdrantVectorStore } from "@langchain/qdrant";


configDotenv()

const llm = new ChatOllama({
    baseUrl: "http://localhost:11434", // Default Ollama URL
    model: "tinyllama", // You can use llama3, mistral, or other models
    temperature: 0,
});


const embeddingModel = new OllamaEmbeddings({
    baseUrl: "http://localhost:11434", // Default Ollama URL
    model: "nomic-embed-text", // A good embedding model in Ollama
});

const qdrantConfig = {
    url: process.env.QDRANT_URL || "http://localhost:6333",
    collectionName: "dt89_collection",
};

export async function langchain(text){
    const path = "./temp.txt";

    await fs.writeFile(path, text);

    const loader = new TextLoader(path);
    const docs = await loader.load();
    
    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 100,
        chunkOverlap: 10,
      });
    const split_text = await splitter.createDocuments([text]);
    
    // Generate embeddings
    const vectorStore = await QdrantVectorStore.fromDocuments(
        split_text,
        embeddingModel,
        qdrantConfig
    );
    console.log("Ingestion Done")
    
}

async function rag(query) {
    const vectorStore = await QdrantVectorStore.fromExistingCollection(
        embeddingModel,
        qdrantConfig
      );

    const results = await vectorStore.similaritySearch(query, 3);
    console.log("Search results:");
    results.forEach((doc, i) => {
        console.log(`Result ${i + 1}:\n`, doc.pageContent);
    });

    const context = results.map((r) => r.pageContent).join("\n");
    const response = await llm.invoke(`Answer this: "${query}" based on the following context:\n${context}`);

    return response.content
}




