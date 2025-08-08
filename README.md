# NECoRT: Nash-Equilibrium Chain of Recursive Thoughts

**NECoRT** is a framework for multi-agent recursive reasoning inspired by game theory. It builds on Chain of Recursive Thought (CoRT) by introducing **multi-agent deliberation** and **Nash equilibrium convergence** to surface robust, self-consistent answers from language models.

---

## 🧠 Concept

| Layer        | Description                                                                 |
|--------------|-----------------------------------------------------------------------------|
| 🧩 CoRT       | A single model recursively evaluates, refines, and verifies its own answers |
| 🤖 NECoRT     | Multiple agents generate and critique responses, converging on consensus    |
| 🎯 Nash Layer | Final response is only accepted if no agent can unilaterally improve it     |

The result is a form of **negotiated truth** — answers that survive scrutiny from multiple angles and stabilize into a response that’s not just correct, but *resilient*.

---

## 🚀 Why It Matters

Traditional LLMs are one-shot, error-prone, and overly confident.

| Method   | Pros                                 | Cons                      |
|----------|--------------------------------------|---------------------------|
| LLM      | Fast                                 | Shallow, brittle answers  |
| CoRT     | Thoughtful, iterative refinement     | Still single perspective  |
| NECoRT   | Diverse, verified, stable reasoning  | Higher compute cost       |

---

## 🔧 How It Works

1. **Agents** propose solutions
2. **Utility Matrix** is built by agents critiquing each other’s answers
3. **Nash Equilibrium** is reached when no agent can unilaterally improve the outcome
4. Final answer is extracted from the most stable agent response

---

## 📁 Project Structure

| File / Folder         | Purpose                                                                 |
|-----------------------|-------------------------------------------------------------------------|
| `nash_recursive_thinking.py` | Core NECoRT logic — agent loop, evaluation, equilibrium checks   |
| `necort_web.py`       | Web interface backend for NECoRT                                         |
| `recthink_web.py`     | Web interface for classic CoRT                                           |
| `frontend/`           | React-based UI                                                           |
| `start-necort.bat`    | Launch NECoRT locally (Windows)                                          |
| `requirements.txt`    | Python dependencies                                                      |
| `rec.PNG`, `non-rec.png` | Visual examples of CoRT vs non-CoRT outputs                          |

---

## 💻 Quickstart

```bash
# Windows
start-necort.bat

# Linux / Mac
pip install -r requirements.txt
cd frontend && npm install && npm start
cd ..
python necort_web.py
```