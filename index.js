import express from 'express'
import bodyParser from 'body-parser';
import { langchain, rag } from './rag.js';

const app = express()
const port = 3000

app.set("view engine", "ejs");
app.use(express.static("public"));
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());

app.get("/", (req, res) => {
    res.render("index");
});

app.post("/chat", async (req, res) => {
  try {
    const context = req.body['context']
    const chatId = Date.now().toString(); 
    if (!context) {
        return res.status(400).json({ error: "Missing context" });
    }
    if (context) {
      await langchain(context, chatId);
    } 
    res.render("chat.ejs",{
      chatId: chatId
    })
    
  } catch (error) {
    console.error("Error in /chat endpoint:", error);
    res.status(500).json({ error: "An error occurred processing your request" });
  }
    
})

app.post("/api/chat", async (req, res) => {
  try {
      const { message, chatId } = req.body;

      if (!message || !chatId) {
          return res.status(400).json({ error: "Missing message or chatId" });
      }

      // Handle the query using RAG (Retrieve and Generate response)
      const response = await rag(message, chatId);

      console.log(response);

      res.json({
          response: response
      });

  } catch (error) {
      console.error("Error in /api/chat endpoint:", error);
      res.status(500).json({ error: "An error occurred during the chat" });
  }
});



app.listen(port, () => {
  console.log(`I'm alive on http://localhost:${port}`)
})
