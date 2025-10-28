const express = require('express');
const app = express();

app.get('/', (req, res) => {
  res.send('Test server works!');
});

const port = 3002;
app.listen(port, () => {
  console.log(`Test server running on http://localhost:${port}`);
});