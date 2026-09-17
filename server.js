require("dotenv").config();
const express = require("express");
const cors = require("cors");
const rateLimit = require("express-rate-limit");
const { GoogleGenAI } = require("@google/genai");

const app = express();
const PORT = process.env.PORT || 3001;

// ── Anthropic client ──────────────────────────────────────────────────────────
const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
});

// ── Middleware ────────────────────────────────────────────────────────────────
app.use(cors({ origin: "http://localhost:5173", credentials: true }));
app.use(express.json());

const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: "Too many requests — try again in a few minutes." },
});
app.use("/api/", limiter);

// ── Health check ──────────────────────────────────────────────────────────────
app.get("/api/health", (req, res) => {
  res.json({ status: "ok", timestamp: new Date().toISOString() });
});

// ── System prompts ────────────────────────────────────────────────────────────
function chatSystemPrompt(topic) {
  return `You are an expert, friendly learning tutor helping a student understand: "${topic}".

Your teaching style:
- Break concepts into clear, digestible steps
- Use real-world analogies and examples
- Check for understanding naturally
- Celebrate curiosity and good questions
- Adjust complexity based on the student's questions
- Use markdown formatting: **bold** for key terms, bullet points for lists, \`code\` for technical terms

Keep responses focused and under 200 words unless the student asks for more depth.
Always end with either a clarifying question or an offer to explore a related concept.`;
}

function quizSystemPrompt(topic) {
  return `You are a quiz generator for the topic: "${topic}".

Generate exactly 4 multiple-choice questions that test genuine understanding (not just memorisation).

Respond ONLY with valid JSON in this exact format — no markdown fences, no extra text:
{
  "questions": [
    {
      "question": "Question text here?",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correctIndex": 0,
      "explanation": "Clear explanation of why this is correct and why others are wrong."
    }
  ]
}

Rules:
- correctIndex is 0-based (0 = first option)
- Make distractors plausible, not obviously wrong
- Explanations should reinforce learning, not just restate the answer
- Vary question difficulty: 1 easy, 2 medium, 1 hard`;
}

function flashcardSystemPrompt(topic, messages) {
  const context =
    messages.length > 0
      ? `\n\nBased on this conversation:\n${messages.slice(-6).map((m) => `${m.role}: ${m.content}`).join("\n")}`
      : "";

  return `You are a flashcard generator for the topic: "${topic}".${context}

Create exactly 5 flashcards targeting the most important concepts from this topic.

Respond ONLY with valid JSON — no markdown fences, no extra text:
{
  "flashcards": [
    {
      "front": "Clear, specific question or prompt",
      "back": "Concise, accurate answer (1-3 sentences max)"
    }
  ]
}

Rules:
- Questions should be specific, not vague
- Answers should be self-contained and memorable
- Cover different aspects: definition, mechanism, example, comparison, application`;
}

function summarySystemPrompt(topic, messages) {
  const history = messages
    .slice(-10)
    .map((m) => `${m.role}: ${m.content}`)
    .join("\n");

  return `Summarise what was learned about "${topic}" in this session.

Conversation:
${history}

Create a structured summary. Respond ONLY with valid JSON:
{
  "summary": {
    "keyPoints": ["point 1", "point 2", "point 3"],
    "conceptsCovered": ["concept A", "concept B"],
    "suggestedNextTopics": ["topic 1", "topic 2"],
    "oneLineSummary": "A single sentence capturing the essence of what was learned."
  }
}`;
}

// ── POST /api/chat  (streaming) ───────────────────────────────────────────────
// ── POST /api/chat (Gemini Streaming) ───────────────────────────────────────
app.post("/api/chat", async (req, res) => {
  const { topic, messages } = req.body;

  if (!topic || !Array.isArray(messages)) {
    return res.status(400).json({
      error: "topic and messages are required."
    });
  }

  // SSE headers
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders();

  try {
    const prompt = `
${chatSystemPrompt(topic)}

Conversation:
${messages.map(m => `${m.role}: ${m.content}`).join("\n")}
`;

    const stream = await ai.models.generateContentStream({
      model: "gemini-3.1-flash-lite",
      contents: prompt,
    });

    for await (const chunk of stream) {
      const text = chunk.text;

      if (text) {
        res.write(
          `data: ${JSON.stringify({
            text: text
          })}\n\n`
        );
      }
    }

    res.write(
      `data: ${JSON.stringify({
        done: true
      })}\n\n`
    );

    res.end();

  } catch (err) {
    console.error("Chat Error:", err);

    res.write(
      `data: ${JSON.stringify({
        error: err.message
      })}\n\n`
    );

    res.end();
  }
});

// ── POST /api/quiz ────────────────────────────────────────────────────────────
app.post("/api/quiz", async (req, res) => {
  const { topic } = req.body;
  if (!topic) return res.status(400).json({ error: "topic is required." });

  try {
   const response = await ai.models.generateContent({
    model: "gemini-3.1-flash-lite",
    contents:
        quizSystemPrompt(topic) +
        "\nGenerate a quiz on " +
        topic
});

const raw = response.text.trim();

const parsed = JSON.parse(raw);

res.json(parsed);

   
  } catch (err) {
    console.error("Quiz error:", err.message);
    res.status(500).json({ error: "Failed to generate quiz. Try again." });
  }
});

// ── POST /api/flashcards ──────────────────────────────────────────────────────
app.post("/api/flashcards", async (req, res) => {
  const { topic, messages = [] } = req.body;
  if (!topic) return res.status(400).json({ error: "topic is required." });

  try {
   const response = await ai.models.generateContent({
    model: "gemini-3.1-flash-lite",
    contents:
        flashcardSystemPrompt(topic, messages)
});

const raw = response.text.trim();

const parsed = JSON.parse(raw);

res.json(parsed);

    
  } catch (err) {
    console.error("Flashcard error:", err.message);
    res.status(500).json({ error: "Failed to generate flashcards. Try again." });
  }
});

// ── POST /api/summary ─────────────────────────────────────────────────────────
app.post("/api/summary", async (req, res) => {
  const { topic, messages = [] } = req.body;
  if (!topic) return res.status(400).json({ error: "topic is required." });

  try {
    const response = await ai.models.generateContent({
    model: "gemini-3.1-flash-lite",
    contents:
        summarySystemPrompt(topic, messages)
});

const raw = response.text.trim();

const parsed = JSON.parse(raw);

res.json(parsed);

    
  } catch (err) {
    console.error("Summary error:", err.message);
    res.status(500).json({ error: "Failed to generate summary. Try again." });
  }
});

// ── Start ─────────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`\n🧠 AI Learning Assistant backend running`);
  console.log(`   http://localhost:${PORT}/api/health\n`);
});

const path = require("path");

app.use(express.static(path.join(__dirname, "dist")));

app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "dist", "index.html"));
});

// For React Router
app.get("*", (req, res) => {
  res.sendFile(path.join(__dirname, "dist", "index.html"));
});