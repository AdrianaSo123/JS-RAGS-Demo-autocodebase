// Simple Express + Socket.IO AI UX chat server
const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const { OpenAI } = require('openai');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
require('dotenv').config();

const app = express();
const server = http.createServer(app);

// Create uploads directory if it doesn't exist
const uploadsDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir);
}

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, uploadsDir);
  },
  filename: function (req, file, cb) {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, 'design-' + uniqueSuffix + path.extname(file.originalname));
  }
});

const upload = multer({
  storage: storage,
  limits: {
    fileSize: 10 * 1024 * 1024 // 10MB limit
  },
  fileFilter: function (req, file, cb) {
    const allowedTypes = /jpeg|jpg|png|gif|webp/;
    const extname = allowedTypes.test(path.extname(file.originalname).toLowerCase());
    const mimetype = allowedTypes.test(file.mimetype);
    
    if (mimetype && extname) {
      return cb(null, true);
    } else {
      cb(new Error('Only image files are allowed!'));
    }
  }
});
const io = new Server(server);

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

app.use(express.static(__dirname + '/public'));
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// Image upload endpoint
app.post('/upload', upload.single('design'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded' });
  }
  
  res.json({
    success: true,
    filename: req.file.filename,
    originalName: req.file.originalname,
    size: req.file.size,
    url: `/uploads/${req.file.filename}`
  });
});

// Function to encode image to base64
function encodeImageToBase64(imagePath) {
  const imageBuffer = fs.readFileSync(imagePath);
  return imageBuffer.toString('base64');
}

// Function to generate UX critique using GPT-4 Vision
async function generateDesignCritique(imagePath, userPrompt = '') {
  try {
    const base64Image = encodeImageToBase64(imagePath);
    
    const systemPrompt = `You are an expert UX designer and usability consultant with extensive experience in user interface design, user research, and accessibility. 

Analyze the uploaded design/wireframe and provide a comprehensive UX critique following this structure:

**🎯 First Impression**
- Overall design clarity and purpose
- Visual hierarchy effectiveness

**📋 Heuristic Evaluation**
Rate each area (1-5 scale, 5 being excellent):
- Visibility of system status
- Match between system and real world  
- User control and freedom
- Consistency and standards
- Error prevention
- Recognition rather than recall
- Flexibility and efficiency of use
- Aesthetic and minimalist design
- Help users recognize, diagnose, and recover from errors
- Help and documentation

**✅ Strengths**
- What works well in this design

**⚠️ Areas for Improvement**
- Specific usability issues identified
- Accessibility concerns

**🚀 Actionable Recommendations**
- Concrete, implementable suggestions
- Priority level for each recommendation

**🎨 Design System Notes**
- Typography, spacing, color observations
- Mobile responsiveness considerations

Keep feedback constructive, specific, and actionable. Reference established UX principles and best practices.`;

    const userMessage = userPrompt ? 
      `Please analyze this design/wireframe. User context: ${userPrompt}` : 
      'Please analyze this design/wireframe and provide detailed UX feedback.';

    const completion = await openai.chat.completions.create({
      model: 'gpt-4-vision-preview',
      messages: [
        {
          role: 'system',
          content: systemPrompt
        },
        {
          role: 'user',
          content: [
            {
              type: 'text',
              text: userMessage
            },
            {
              type: 'image_url',
              image_url: {
                url: `data:image/jpeg;base64,${base64Image}`,
                detail: 'high'
              }
            }
          ]
        }
      ],
      max_tokens: 1500,
      temperature: 0.3
    });

    return completion.choices[0].message.content;
  } catch (error) {
    console.error('Error generating design critique:', error);
    throw error;
  }
}

io.on('connection', (socket) => {
  socket.on('user message', async (msg) => {
    // Compose prompt for AI UX professional
    let aiResponse = 'Sorry, I could not generate a response.';
    try {
      const completion = await openai.chat.completions.create({
        model: 'gpt-3.5-turbo',
        messages: [
          { role: 'system', content: 'You are a kind, knowledgeable, and professional UX consultant, similar to someone who works for the Nielsen Norman Group. Your responses must be short and to the point (2-4 sentences max, or a brief bullet list). Be clear, concise, and friendly. Avoid long explanations.' },
          { role: 'user', content: msg }
        ]
      });
      aiResponse = completion.choices[0].message.content;
    } catch (e) {
      aiResponse = 'Error: ' + e.message;
    }
    socket.emit('ai message', aiResponse);
  });

  // Handle design critique requests
  socket.on('analyze design', async (data) => {
    const { filename, userPrompt } = data;
    const imagePath = path.join(uploadsDir, filename);
    
    if (!fs.existsSync(imagePath)) {
      socket.emit('critique error', 'Image file not found');
      return;
    }

    try {
      const critique = await generateDesignCritique(imagePath, userPrompt);
      socket.emit('design critique', {
        critique: critique,
        filename: filename,
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      console.error('Design analysis error:', error);
      socket.emit('critique error', 'Failed to analyze design. Please try again.');
    }
  });
});

const PORT = process.env.PORT || 3002;
server.listen(PORT, () => {
  console.log(`AI UX Chat server listening on port ${PORT}`);
});
