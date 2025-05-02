# AI UX Chat Demo

A real-time chat demo with an AI UX professional using Node.js, Express, Socket.IO, and OpenAI.

## Features
- Real-time chat interface styled to match portfolio branding
- AI-powered UX advice via OpenAI API
- Modern, accessible, and mobile-friendly UI

## Getting Started

1. **Install dependencies:**
   ```bash
   npm install
   ```
2. **Create a `.env` file:**
   ```bash
   cp .env.example .env
   # Add your OpenAI API key to .env
   ```
3. **Start the server:**
   ```bash
   npm start
   ```
4. **Visit:**
   [http://localhost:3002](http://localhost:3002)

## Deployment (Render)
- Push this folder to a GitHub repository.
- Create a new Web Service on [Render](https://render.com/).
- Set the environment variable `OPENAI_API_KEY`.
- Use `node server.js` as the start command.

## Environment Variables
- `OPENAI_API_KEY` – Your OpenAI API key for chat completion.

---
MIT License
