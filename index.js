import express from 'express'
import bodyParser from 'body-parser';
import { langchain } from './rag.js';

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
    const context = req.body['context']
    const chatId = Date.now().toString(); 
    await langchain(context)
})

app.listen(port, () => {
  console.log(`I'm alive on http://localhost:${port}`)
})
