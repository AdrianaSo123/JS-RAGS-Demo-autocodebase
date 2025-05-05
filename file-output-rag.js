// RAG Demo with LangChain and OpenAI - with file output
import { config } from 'dotenv';
import { DirectoryLoader } from 'langchain/document_loaders/fs/directory';
import { TextLoader } from 'langchain/document_loaders/fs/text';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { OpenAIEmbeddings } from '@langchain/openai';
import { ChatOpenAI } from '@langchain/openai';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import fs from 'fs/promises';

// Load environment variables
config();

// Verify we have an OpenAI API key
if (!process.env.OPENAI_API_KEY) {
  console.error('Error: OPENAI_API_KEY is not set in the environment variables.');
  console.error('Please make sure you have a .env file with your OpenAI API key.');
  process.exit(1);
}

// Create a log file for output
async function logToFile(content) {
  await fs.appendFile('rag-output.txt', content + '\n');
  console.log(content);
}

// Main function to demonstrate RAG
async function runRAGDemo() {
  // Reset the log file
  await fs.writeFile('rag-output.txt', '');
  
  await logToFile('🚀 Starting RAG Demo with LangChain and OpenAI');
  
  try {
    // Initialize models with explicit configuration
    const embeddings = new OpenAIEmbeddings({
      modelName: "text-embedding-ada-002",
      stripNewLines: true
    });
    
    const llm = new ChatOpenAI({
      modelName: 'gpt-3.5-turbo',
      temperature: 0.2
    });
    
    // Step 1: Load documents
    await logToFile('\n📚 Loading documents...');
    const loader = new DirectoryLoader('./documents', {
      '.txt': (path) => new TextLoader(path)
    });
    
    const docs = await loader.load();
    await logToFile(`Loaded ${docs.length} documents.`);
    
    // Step 2: Split documents into chunks
    await logToFile('\n✂️ Splitting documents into chunks...');
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 500,
      chunkOverlap: 100
    });
    
    const splitDocs = await textSplitter.splitDocuments(docs);
    await logToFile(`Created ${splitDocs.length} chunks.`);
    
    // Check document validity
    for (let i = 0; i < splitDocs.length; i++) {
      if (typeof splitDocs[i].pageContent !== 'string') {
        await logToFile(`Document at index ${i} has invalid pageContent type: ${typeof splitDocs[i].pageContent}`);
        splitDocs[i].pageContent = String(splitDocs[i].pageContent || '');
      }
    }
    
    // Step 3: Create vector store
    await logToFile('\n🧠 Creating in-memory vector store...');
    const vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, embeddings);
    await logToFile('Vector store created successfully.');
    
    // Handle queries directly without complex chains
    const queries = [
      'What is RAG and what are its key components?',
      'What are vector databases and how do they work?',
      'How does embedding generation work in RAG systems?'
    ];
    
    for (const query of queries) {
      await processQuery(query, vectorStore, llm);
    }
    
  } catch (error) {
    await logToFile(`Error in RAG demo: ${error.message}`);
    await logToFile(error.stack);
  }
}

// Process a single query using the vector store
async function processQuery(query, vectorStore, llm) {
  await logToFile(`\n❓ Question: ${query}`);
  await logToFile('Retrieving relevant documents...');
  
  try {
    // Get similar documents
    const relevantDocs = await vectorStore.similaritySearch(query, 2);
    await logToFile(`Found ${relevantDocs.length} relevant documents.`);
    
    // Extract context
    const context = relevantDocs.map((doc, i) => {
      return `Document ${i+1}:\n${doc.pageContent}`;
    }).join('\n\n');
    
    // Generate response
    await logToFile('Generating answer using context...');
    const prompt = `
      Answer the question based on the following context:
      
      ${context}
      
      Question: ${query}
      
      When answering, make sure to cite your source as [Document X] where X is the document number.
    `;
    
    const response = await llm.invoke(prompt);
    await logToFile('\n🔍 Answer:');
    await logToFile(response.content);
    
  } catch (error) {
    await logToFile(`Error processing query: ${error.message}`);
    await logToFile(error.stack);
  }
}

// Run the demo
runRAGDemo()
  .then(() => logToFile('\nRAG demo completed.'))
  .catch(error => logToFile(`Fatal error in RAG demo: ${error.message}`));
