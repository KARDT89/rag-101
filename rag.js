import { TextLoader } from "langchain/document_loaders/fs/text";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import fs from "fs/promises";
import { configDotenv } from "dotenv";
import { ChatOllama, OllamaEmbeddings } from "@langchain/ollama";
import { QdrantVectorStore } from "@langchain/qdrant";


configDotenv()

const llm = new ChatOllama({
    baseUrl: "http://localhost:11434", 
    model: "gemma:2b", 
    temperature: 0,
});


const embeddingModel = new OllamaEmbeddings({
    baseUrl: "http://localhost:11434", 
    model: "nomic-embed-text", 
});


export async function langchain(text, chatId) {
    try {
        const path = "./temp.txt";
        await fs.writeFile(path, text);

        const loader = new TextLoader(path); // Load the file
        const docs = await loader.load();
        
        const splitter = new RecursiveCharacterTextSplitter({
            chunkSize: 500, // Increased chunk size for better context
            chunkOverlap: 50, // Increased overlap to maintain context between chunks
        });
        const split_text = await splitter.splitDocuments(docs)

        const qdrantConfig = {
            url: process.env.QDRANT_URL || "http://localhost:6333",
            collectionName: `letsgo-${chatId}`,
        };
        
        // Generate embeddings and store in Qdrant
        await QdrantVectorStore.fromDocuments(
            split_text,
            embeddingModel,
            qdrantConfig
        );
        
        
        console.log(`Ingestion complete: ${split_text.length} chunks processed`);
        return { success: true, chunks: split_text.length };

    } catch (error) {
        console.error("Error during ingestion:", error);
        throw error;
    }
}

export async function rag(query, chatId) {
    try{
        const qdrantConfig = {
            url: process.env.QDRANT_URL || "http://localhost:6333",
            collectionName: `letsgo-${chatId}`,
        };

        const vectorStore = await QdrantVectorStore.fromExistingCollection(
            embeddingModel,
            qdrantConfig
          );
    
        const results = await vectorStore.similaritySearch(query, 5);
        
        console.log("Search results:");
        results.forEach((doc, i) => {
            console.log(`Result ${i + 1}:\n`, doc.pageContent);
        });
    
        const context = results.map((r) => r.pageContent).join("\n");
        
        const prompt = `You are a helpful AI assistant with deep analytical capabilities.

                        CONTEXT INFORMATION:
                        """
                        ${context}
                        """

                        USER QUERY: "${query}"

                        Instructions:
                        0. Format it for Professional use.
                        1. Analyze the context carefully and extract relevant information to address the query
                        2. Keep it simple and to the point.
                        3. Add humor to your answers ocationally.
                        4. If you dont have context to the answer just say "I dont have context". Don't hallucinate.
                        5. Write a detailed, professional, and well-structured response on ${query}, 
                        formatted with clean Markdown. The tone should be expert yet approachable, 
                        Occasionally (but sparingly), include well-placed emojis to enhance engagement without undermining the professional tone. 
                        Avoid overly casual language, but keep it human.

                        Remember: Quality over quantity. Precision is more important than length.

                        Example: 
                        Context: "My name is Tamal Sarkar! I love to code and play videogames!"

                        User query: "Who's Better Messi or Ronaldo?"
                        response: "I dont have context to answer that"
                        
                        User query: "Hi chat!"
                        response: "Hey Tamal! How can I help you?"
                        `;
        
        const response = await llm.invoke(prompt);

        return response.content

    } catch (error) {
        console.error("Error during RAG:", error);
        throw error;
    }
    
}




