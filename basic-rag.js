// Modified version of basic-rag.js that uses BERT for embeddings and OpenAI for generation

import { ChromaClient } from 'chromadb';
import { PDFLoader } from 'langchain/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import * as tf from '@tensorflow/tfjs-node'; // Add TensorFlow.js
import * as tokenizers from '@nlpjs/bert-tokenizer'; // Add BERT tokenizer
import OpenAI from 'openai';
import { config } from 'dotenv';
import * as fs from 'fs';
import * as path from 'path';
import readline from 'readline';

// Load environment variables (including your OpenAI API key)
config();

// Initialize OpenAI client (still needed for generation/completion)
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Initialize ChromaDB client
const client = new ChromaClient();

// Function to load BERT model (new addition)
async function loadBertModel() {
  console.log("Loading BERT model...");
  const modelPath = './bert_model/model.json'; // Path to your converted BERT model
  const model = await tf.loadGraphModel(modelPath);
  return model;
}

// Function to get BERT embeddings (new addition)
async function getBertEmbeddings(text, model, bertTokenizer) {
  // Tokenize the text using BERT tokenizer
  const tokenized = bertTokenizer.tokenize(text);
  const inputIds = tokenized.ids.slice(0, 512); // Truncate to BERT's max length
  const attentionMask = new Array(inputIds.length).fill(1);
  
  // Convert to tensors
  const inputTensor = tf.tensor2d([inputIds], [1, inputIds.length]);
  const maskTensor = tf.tensor2d([attentionMask], [1, attentionMask.length]);
  
  // Get embeddings from the model
  const outputs = model.execute({
    input_ids: inputTensor,
    attention_mask: maskTensor
  });
  
  // Get the CLS token embedding (first token) as the sentence embedding
  const embeddings = outputs.last_hidden_state.slice([0, 0, 0], [1, 1, -1]).dataSync();
  
  // Clean up tensors
  tf.dispose([inputTensor, maskTensor, outputs]);
  
  return Array.from(embeddings);
}

// Initialize RAG system (mostly the same, but uses BERT for embeddings)
async function initializeRag() {
  try {
    // Load BERT model and tokenizer (new addition)
    const bertModel = await loadBertModel();
    const bertTokenizer = new tokenizers.BertTokenizer();
    await bertTokenizer.load();
    
    // Create or get a collection (same as before)
    let collection = await client.getOrCreateCollection({
      name: "rag_collection",
      metadata: { hnsw:configuration: { M: 16, efConstruction: 200 } }
    });

    // Process PDF documents (same as before)
    const pdfPath = path.resolve('./docs/sample.pdf');
    console.log(`Loading PDF from ${pdfPath}...`);
    const loader = new PDFLoader(pdfPath);
    const docs = await loader.load();
    
    // Split documents into chunks (same as before)
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });
    const chunks = await textSplitter.splitDocuments(docs);
    console.log(`Split into ${chunks.length} chunks`);
    
    // Get embeddings for each chunk using BERT (modified to use BERT)
    console.log("Getting embeddings for chunks...");
    const embeddings = [];
    const texts = [];
    const metadatas = [];
    const ids = [];
    
    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      // CHANGED: Use BERT for embeddings instead of OpenAI
      const embedding = await getBertEmbeddings(chunk.pageContent, bertModel, bertTokenizer);
      embeddings.push(embedding);
      texts.push(chunk.pageContent);
      metadatas.push(chunk.metadata);
      ids.push(`id${i}`);
    }
    
    // Add documents to the collection (same as before)
    await collection.add({
      ids: ids,
      embeddings: embeddings,
      metadatas: metadatas,
      documents: texts,
    });
    
    console.log("Documents added to collection");
    
    // Return the initialized components for querying
    return { bertModel, bertTokenizer, collection };
  } catch (error) {
    console.error("Error initializing RAG:", error);
    throw error;
  }
}

// Query the RAG system (mostly the same, but uses BERT for query embedding)
async function queryRag(query, { bertModel, bertTokenizer, collection }) {
  try {
    // CHANGED: Get embedding for the query using BERT instead of OpenAI
    const queryEmbedding = await getBertEmbeddings(query, bertModel, bertTokenizer);
    
    // Query the collection (same as before)
    const results = await collection.query({
      queryEmbeddings: [queryEmbedding],
      nResults: 5,
    });
    
    // Extract the most relevant chunks (same as before)
    const relevantChunks = results.documents[0];
    
    // Construct a prompt with the retrieved contexts (same as before)
    const prompt = `
You are a helpful assistant that answers questions based on the provided context.

Context:
${relevantChunks.join("\n\n")}

Question: ${query}

Answer:
`;
    
    // Use OpenAI for the completion/generation part (same as before)
    const completion = await openai.chat.completions.create({
      model: "gpt-3.5-turbo",
      messages: [{ role: "user", content: prompt }],
      max_tokens: 500,
      temperature: 0.7,
    });
    
    return {
      answer: completion.choices[0].message.content,
      context: relevantChunks.join("\n\n"),
      relevantChunks: relevantChunks
    };
  } catch (error) {
    console.error("Error querying RAG:", error);
    throw error;
  }
}

// Main execution (mostly the same, but pass BERT components to queryRag)
async function main() {
  const { bertModel, bertTokenizer, collection } = await initializeRag();
  
  // Create readline interface for user input (same as before)
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
  });
  
  // Process user queries in a loop (same as before)
  const askQuestion = () => {
    rl.question('Enter your question (or "exit" to quit): ', async (query) => {
      if (query.toLowerCase() === 'exit') {
        rl.close();
        return;
      }
      
      try {
        // Query the RAG system (pass BERT components)
        const result = await queryRag(query, { bertModel, bertTokenizer, collection });
        
        // Output the result (same as before)
        console.log("\nAnswer:", result.answer);
        console.log("\n---\n");
        
        // Ask for the next question (same as before)
        askQuestion();
      } catch (error) {
        console.error("Error processing query:", error);
        // Ask for the next question (same as before)
        askQuestion();
      }
    });
  };
  
  // Start the question-answering loop (same as before)
  console.log("RAG system initialized. You can now ask questions.");
  askQuestion();
}

// Run the main function (same as before)
main().catch(console.error);
