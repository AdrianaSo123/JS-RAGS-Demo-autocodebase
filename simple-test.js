// Simple diagnostic test for OpenAI embeddings and RAG
import { config } from 'dotenv';
import { OpenAIEmbeddings } from '@langchain/openai';
import { ChatOpenAI } from '@langchain/openai';

// Load environment variables
config();

// Verify we have an OpenAI API key
if (!process.env.OPENAI_API_KEY) {
  console.error('Error: OPENAI_API_KEY is not set in the environment variables.');
  process.exit(1);
}

async function testOpenAI() {
  try {
    console.log('Starting OpenAI API test...');
    
    // Initialize embeddings with explicit configuration
    const embeddings = new OpenAIEmbeddings({
      modelName: "text-embedding-ada-002",
      stripNewLines: true
    });
    
    // Test embeddings
    console.log('Testing OpenAI embeddings...');
    const testText = 'This is a test sentence for embeddings.';
    try {
      const embedding = await embeddings.embedQuery(testText);
      console.log(`Successfully generated embedding with ${embedding.length} dimensions`);
    } catch (error) {
      console.error('Error generating embeddings:', error);
    }
    
    // Test chat completions
    console.log('\nTesting OpenAI chat completions...');
    const llm = new ChatOpenAI({
      modelName: 'gpt-3.5-turbo',
      temperature: 0.2
    });
    
    try {
      const result = await llm.invoke("What is RAG in the context of AI?");
      console.log('Chat completion result:');
      console.log(result.content);
    } catch (error) {
      console.error('Error generating chat completion:', error);
    }
    
  } catch (error) {
    console.error('Test failed with error:', error);
  }
}

// Run test
testOpenAI()
  .then(() => console.log('\nTest completed.'))
  .catch(err => console.error('Unhandled error:', err));
