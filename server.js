// Simple Express + Socket.IO AI UX chat server
const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const { OpenAI } = require('openai');
require('dotenv').config();

const app = express();
const server = http.createServer(app);
const io = new Server(server);

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

app.use(express.static(__dirname + '/public'));

io.on('connection', (socket) => {
  socket.on('user message', async (msg) => {
    // Compose prompt for AI UX professional
    const prompt = `You are a helpful, expert UX professional. Respond to the following user message with practical, empathetic, and actionable UX advice.\nUser: ${msg}\nAI UX Professional:`;
    let aiResponse = 'Sorry, I could not generate a response.';
    try {
      const completion = await openai.chat.completions.create({
        model: 'gpt-3.5-turbo',
        messages: [
          { role: 'system', content: 'You are a kind, knowledgeable, and professional UX consultant, similar to someone who works for the Nielsen Norman Group. Your responses must be short and to the point (2-4 sentences max, or a brief bullet list). Be clear, concise, and friendly. Avoid long explanations.' },,
          { role: 'user', content: msg }
        ]
      });
      aiResponse = completion.choices[0].message.content;
    } catch (e) {
      aiResponse = 'Error: ' + e.message;
    }
    socket.emit('ai message', aiResponse);
  });
});

const PORT = process.env.PORT || 3002;
server.listen(PORT, () => {
  console.log(`AI UX Chat server listening on port ${PORT}`);
});
