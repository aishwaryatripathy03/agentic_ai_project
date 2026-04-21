"""
agent.py
Study Buddy — Physics Assistant
Domain: B.Tech Physics (1st/2nd Year)
Agentic AI with RAG, Memory, Tools, LangGraph, Self-Evaluation
"""

import os
import math
from datetime import datetime
from typing import TypedDict, List

import chromadb
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END

# ─────────────────────────────────────────────
# 1. KNOWLEDGE BASE — 10 Physics Documents
# ─────────────────────────────────────────────
documents = [
    {
        "id": "doc_001",
        "topic": "Kinematics Basics",
        "text": (
            "Kinematics is the branch of physics that studies the motion of objects without considering "
            "the causes of motion. It deals with quantities such as displacement, velocity, and acceleration. "
            "Displacement is the shortest distance between initial and final position and is a vector quantity, "
            "while distance is the total path covered and is scalar. Velocity is the rate of change of "
            "displacement and has both magnitude and direction, whereas speed is scalar. "
            "Acceleration is the rate of change of velocity. A body moving with uniform velocity has zero "
            "acceleration. Kinematics helps describe motion using equations and graphs. "
            "Position-time graphs show displacement vs time (slope = velocity), while velocity-time graphs "
            "show velocity vs time (slope = acceleration, area = displacement). "
            "These tools are essential for understanding more advanced topics in mechanics."
        ),
    },
    {
        "id": "doc_002",
        "topic": "Equations of Motion",
        "text": (
            "The equations of motion describe the relationship between velocity, acceleration, time, and "
            "displacement for uniformly accelerated motion. The three main equations are: "
            "1) v = u + at, where u is initial velocity, v is final velocity, a is acceleration, t is time. "
            "2) s = ut + (1/2)at^2, where s is displacement. "
            "3) v^2 = u^2 + 2as. "
            "These equations are valid only when acceleration is constant (uniform acceleration). "
            "For free fall, acceleration a = g = 9.8 m/s^2 downward. "
            "Example: A car starts from rest (u=0) and accelerates at 2 m/s^2 for 5 seconds. "
            "Final velocity v = 0 + 2 x 5 = 10 m/s. Distance s = 0 x 5 + (1/2) x 2 x 25 = 25 m. "
            "These equations are fundamental for solving numerical problems in linear motion."
        ),
    },
    {
        "id": "doc_003",
        "topic": "Newton's First Law",
        "text": (
            "Newton's First Law of Motion states that an object will remain at rest or in uniform motion "
            "in a straight line unless acted upon by an external net force. "
            "This law introduces the concept of inertia, which is the tendency of an object to resist "
            "changes in its state of motion. Heavier objects have greater inertia. "
            "Examples: A stationary ball will not move unless a force is applied. "
            "A moving bus passenger falls forward when brakes are applied because the body tends to continue moving. "
            "Seatbelts are designed based on this principle. "
            "This law is also called the Law of Inertia. "
            "If the net force on a body is zero, it may still be moving with constant velocity."
        ),
    },
    {
        "id": "doc_004",
        "topic": "Newton's Second Law",
        "text": (
            "Newton's Second Law of Motion states that the net force acting on an object equals the product "
            "of its mass and acceleration: F = ma. "
            "Force is measured in Newtons (N). 1 N = 1 kg x m/s^2. "
            "A larger force produces a greater acceleration for the same mass. "
            "A larger mass results in smaller acceleration for the same force. "
            "Example 1: A 5 kg object accelerates at 3 m/s^2. Force = 5 x 3 = 15 N. "
            "Example 2: A 10 N force acts on a 2 kg object. Acceleration = 10/2 = 5 m/s^2. "
            "This law also defines momentum p = mv, and F = dp/dt (rate of change of momentum). "
            "Impulse = F x t = change in momentum. This law is the cornerstone of dynamics."
        ),
    },
    {
        "id": "doc_005",
        "topic": "Work Energy Power",
        "text": (
            "Work is done when a force displaces an object in the direction of force. "
            "Formula: W = F x d x cos(theta), where theta is the angle between force and displacement. "
            "Unit of work is Joule (J). If theta = 90 degrees, work done = 0. "
            "Kinetic Energy (KE) = (1/2)mv^2. It is the energy of motion. "
            "Potential Energy (PE) = mgh. It is stored energy due to position. "
            "Work-Energy Theorem: Net work done on an object = change in its kinetic energy. W = delta(KE). "
            "Law of Conservation of Energy: Total mechanical energy (KE + PE) remains constant "
            "in the absence of non-conservative forces. "
            "Power is the rate of doing work: P = W/t. Unit is Watt (W). 1 Watt = 1 J/s. "
            "Example: A machine does 500 J of work in 10 s. Power = 500/10 = 50 W."
        ),
    },
    {
        "id": "doc_006",
        "topic": "Circular Motion",
        "text": (
            "Circular motion is the movement of an object along a circular path. "
            "In uniform circular motion, the speed remains constant but the direction changes continuously, "
            "resulting in centripetal acceleration directed towards the center: a = v^2/r. "
            "Centripetal force: F = mv^2/r = m x omega^2 x r. "
            "Angular velocity omega = v/r, measured in rad/s. "
            "Time period T = 2 x pi x r / v = 2 x pi / omega. Frequency f = 1/T. "
            "Examples: satellite orbiting Earth, stone on a string, car on a curved road. "
            "Centripetal force is not a new force — it is provided by existing forces like tension, "
            "gravity, or friction. Centrifugal force is a pseudo-force felt in rotating frames."
        ),
    },
    {
        "id": "doc_007",
        "topic": "Gravitation",
        "text": (
            "Gravitation is the universal force of attraction between any two masses. "
            "Newton's Law: F = G x m1 x m2 / r^2, where G = 6.674 x 10^-11 N m^2/kg^2. "
            "Acceleration due to gravity on Earth: g = GM/R^2 approximately 9.8 m/s^2. "
            "g decreases with altitude: g' = g(1 - 2h/R) for small h. "
            "Escape velocity = sqrt(2gR) approximately 11.2 km/s. "
            "Orbital velocity for a satellite: v = sqrt(GM/r). "
            "Kepler's Third Law: T^2 is proportional to r^3. "
            "Gravitational potential energy: U = -GMm/r."
        ),
    },
    {
        "id": "doc_008",
        "topic": "Simple Harmonic Motion",
        "text": (
            "Simple Harmonic Motion (SHM) is an oscillatory motion where the restoring force is "
            "directly proportional to displacement and acts opposite to it: F = -kx. "
            "Equation: x(t) = A sin(omega x t + phi), A is amplitude, omega is angular frequency. "
            "For spring-mass system: omega = sqrt(k/m), T = 2 x pi x sqrt(m/k). "
            "For simple pendulum: T = 2 x pi x sqrt(L/g). "
            "At mean position: velocity is maximum (A x omega), acceleration = 0. "
            "At extreme position: velocity = 0, acceleration is maximum (A x omega^2). "
            "Total energy in SHM = (1/2) x k x A^2 = constant. "
            "Examples: mass on spring, simple pendulum (small angles), vibrating tuning fork."
        ),
    },
    {
        "id": "doc_009",
        "topic": "Waves",
        "text": (
            "A wave is a disturbance that transfers energy without transferring matter. "
            "Mechanical waves need a medium (e.g., sound, water waves). "
            "Electromagnetic waves need no medium (e.g., light, radio waves). "
            "Transverse waves: oscillation perpendicular to direction (e.g., light). "
            "Longitudinal waves: oscillation parallel to direction (e.g., sound). "
            "Wavelength (lambda) = distance between two consecutive crests. "
            "Frequency (f) = number of oscillations per second (Hz). "
            "Wave speed: v = f x lambda. "
            "Speed of sound in air approximately 340 m/s. Speed of light = 3 x 10^8 m/s. "
            "Superposition: waves add algebraically when they meet. "
            "Constructive interference: waves in phase, amplitude doubles. "
            "Destructive interference: waves out of phase, amplitude cancels. "
            "Standing waves form when incident and reflected waves superpose."
        ),
    },
    {
        "id": "doc_010",
        "topic": "Units and Dimensions",
        "text": (
            "Physical quantities are measured using units. The SI system defines 7 base units: "
            "meter (m) for length, kilogram (kg) for mass, second (s) for time, "
            "Ampere (A) for current, Kelvin (K) for temperature, mole (mol) for amount, "
            "candela (cd) for luminous intensity. "
            "Dimensions express quantities in base terms: "
            "Velocity = [L T^-1], Acceleration = [L T^-2], Force = [M L T^-2], Energy = [M L^2 T^-2]. "
            "Dimensional analysis: check correctness of equations, derive relationships, convert units. "
            "Limitation: cannot determine dimensionless constants like 1/2, pi. "
            "Significant figures maintain precision in measurements and calculations."
        ),
    },
]


# ─────────────────────────────────────────────
# 2. CHROMADB SETUP
# ─────────────────────────────────────────────
def setup_chromadb():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.Client()
    try:
        collection = client.create_collection(name="study_buddy_physics")
    except Exception:
        client.delete_collection("study_buddy_physics")
        collection = client.create_collection(name="study_buddy_physics")

    texts = [doc["text"] for doc in documents]
    embeddings = model.encode(texts).tolist()
    collection.add(
        documents=texts,
        metadatas=[{"topic": doc["topic"]} for doc in documents],
        ids=[doc["id"] for doc in documents],
        embeddings=embeddings,
    )
    return collection, model


# ─────────────────────────────────────────────
# 3. STATE
# ─────────────────────────────────────────────
class CapstoneState(TypedDict):
    question: str
    messages: List[str]
    route: str
    retrieved: str
    sources: List[str]
    tool_result: str
    answer: str
    faithfulness: float
    eval_retries: int
    user_name: str


# ─────────────────────────────────────────────
# 4. TOOLS
# ─────────────────────────────────────────────
def calculator_tool(expression: str) -> str:
    try:
        allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        allowed.update({"abs": abs, "round": round, "pow": pow})
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"Result: {round(result, 6)}"
    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as e:
        return f"Calculator error: {str(e)}"


def datetime_tool() -> str:
    try:
        return datetime.now().strftime("%A, %d %B %Y — %I:%M %p")
    except Exception as e:
        return f"Error: {str(e)}"


# ─────────────────────────────────────────────
# 5. LLM (Groq)
# ─────────────────────────────────────────────
import os
from groq import Groq

os.environ["GROQ_API_KEY"] = "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"
client_llm = Groq(api_key=os.getenv("GROQ_API_KEY"))

def call_llm(prompt: str, system: str = "") -> str:
    try:
        messages = []
        
        if system:
            messages.append({
                "role": "system",
                "content": system
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })

        response = client_llm.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0,
            max_tokens=600
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"LLM Error: {str(e)}"


# ─────────────────────────────────────────────
# 6. NODES
# ─────────────────────────────────────────────
_collection = None
_embed_model = None


def memory_node(state: CapstoneState) -> CapstoneState:
    question = state.get("question", "")
    messages = state.get("messages", [])
    user_name = state.get("user_name", "")
    q_lower = question.lower()
    for phrase in ["my name is", "i am", "i'm", "call me"]:
        if phrase in q_lower:
            parts = q_lower.split(phrase)
            if len(parts) > 1:
                candidate = parts[1].strip().split()[0].capitalize()
                if candidate.isalpha():
                    user_name = candidate
                    break
    messages.append(f"Student: {question}")
    return {**state, "messages": messages, "user_name": user_name}


def router_node(state: CapstoneState) -> CapstoneState:
    question = state.get("question", "")
    messages = state.get("messages", [])
    history = "\n".join(messages[-6:])
    system_prompt = (
        "You are a routing agent for a B.Tech physics study bot. "
        "Decide the best action.\n"
        "- Physics concept/formula/law/topic → return: retrieve\n"
        "- Calculate or solve a numerical → return: tool\n"
        "- Date/time question → return: tool\n"
        "- Student sharing name, greeting → return: skip\n"
        "- Out of scope (politics, cricket, etc.) → return: skip\n"
        "Return ONLY one word: retrieve / tool / skip"
    )
    route = call_llm(
        f"Conversation:\n{history}\n\nQuestion: {question}", system=system_prompt
    ).lower().strip()
    if route not in ["retrieve", "tool", "skip"]:
        route = "retrieve"
    return {**state, "route": route}


def retrieval_node(state: CapstoneState) -> CapstoneState:
    question = state.get("question", "")
    q_embed = _embed_model.encode([question]).tolist()
    results = _collection.query(query_embeddings=q_embed, n_results=3)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    return {**state, "retrieved": "\n\n".join(docs),
            "sources": [m.get("topic", "Unknown") for m in metas]}


def tool_node(state: CapstoneState) -> CapstoneState:
    question = state.get("question", "")
    q_lower = question.lower()
    if any(w in q_lower for w in ["date", "time", "today", "day"]):
        result = datetime_tool()
    else:
        expr = call_llm(
            f"Extract ONLY the Python math expression from: '{question}'. "
            "Return only the expression, nothing else."
        )
        result = f"Expression: {expr} → {calculator_tool(expr.strip())}"
    return {**state, "tool_result": result, "retrieved": "", "sources": []}


def skip_node(state: CapstoneState) -> CapstoneState:
    return {**state, "retrieved": "", "sources": [], "tool_result": ""}


def answer_node(state: CapstoneState) -> CapstoneState:
    question = state.get("question", "")
    retrieved = state.get("retrieved", "")
    tool_result = state.get("tool_result", "")
    route = state.get("route", "skip")
    user_name = state.get("user_name", "")
    messages = state.get("messages", [])
    name_str = f" The student's name is {user_name}." if user_name else ""

    system_prompt = (
        "You are PhysicsBot, a friendly study assistant for B.Tech physics students.\n"
        "RULES:\n"
        "1. Answer ONLY from the provided context. Never hallucinate formulas.\n"
        "2. If not in context: 'I don't have info on that. Check your textbook.'\n"
        "3. For concepts: definition → formula → example.\n"
        "4. For numericals: show step-by-step solution.\n"
        "5. Do NOT answer out-of-scope topics.\n"
        f"{name_str}"
    )

    if route == "tool" and tool_result:
        prompt = f"Student asked: '{question}'\nTool output: {tool_result}\nGive a clear friendly response."
    elif route == "retrieve" and retrieved:
        prompt = (
            f"Physics Context:\n{retrieved}\n\n"
            f"History:\n" + "\n".join(messages[-4:]) +
            f"\n\nQuestion: {question}\nAnswer using ONLY the context."
        )
    else:
        if "name" in question.lower() and user_name:
            prompt = f"Student's name is {user_name}. Greet them warmly as PhysicsBot."
        elif any(g in question.lower() for g in ["hello", "hi", "hey"]):
            prompt = f"Student said: '{question}'. Greet as PhysicsBot and offer physics help."
        else:
            prompt = f"'{question}' is out of scope. Politely decline and say you only help with B.Tech physics."

    return {**state, "answer": call_llm(prompt, system=system_prompt)}


def eval_node(state: CapstoneState) -> CapstoneState:
    answer = state.get("answer", "")
    retrieved = state.get("retrieved", "")
    route = state.get("route", "skip")
    retries = state.get("eval_retries", 0)
    if route != "retrieve" or not retrieved:
        return {**state, "faithfulness": 1.0, "eval_retries": retries}
    score_str = call_llm(
        f"Context:\n{retrieved}\n\nAnswer:\n{answer}\n\nScore (0–1):",
        system="Score how grounded the answer is in the context. Return ONLY a number 0–1."
    )
    try:
        score = max(0.0, min(1.0, float(score_str.strip())))
    except Exception:
        score = 0.5
    return {**state, "faithfulness": score, "eval_retries": retries}


def save_node(state: CapstoneState) -> CapstoneState:
    messages = state.get("messages", [])
    messages.append(f"PhysicsBot: {state.get('answer', '')}")
    return {**state, "messages": messages}


def retry_node(state: CapstoneState) -> CapstoneState:
    retries = state.get("eval_retries", 0) + 1
    state = {**state, "question": f"Detailed explanation: {state.get('question', '')}", "eval_retries": retries}
    state = retrieval_node(state)
    return answer_node(state)


# ─────────────────────────────────────────────
# 7. GRAPH
# ─────────────────────────────────────────────
def route_decision(state): return state.get("route", "skip")
def eval_decision(state):
    return "retry" if state.get("faithfulness", 1.0) < 0.5 and state.get("eval_retries", 0) < 2 else "save"


def build_graph(collection, embed_model):
    global _collection, _embed_model
    _collection = collection
    _embed_model = embed_model

    g = StateGraph(CapstoneState)
    for name, fn in [("memory", memory_node), ("router", router_node), ("retrieval", retrieval_node),
                     ("tool", tool_node), ("skip", skip_node), ("answer", answer_node),
                     ("eval", eval_node), ("save", save_node), ("retry", retry_node)]:
        g.add_node(name, fn)

    g.set_entry_point("memory")
    g.add_edge("memory", "router")
    g.add_conditional_edges("router", route_decision, {"retrieve": "retrieval", "tool": "tool", "skip": "skip"})
    for n in ["retrieval", "tool", "skip"]:
        g.add_edge(n, "answer")
    g.add_edge("answer", "eval")
    g.add_conditional_edges("eval", eval_decision, {"retry": "retry", "save": "save"})
    g.add_edge("retry", "save")
    g.add_edge("save", END)
    return g.compile()


def initialize():
    collection, embed_model = setup_chromadb()
    return build_graph(collection, embed_model)


def run_query(app, question: str, state: dict = None) -> dict:
    if state is None:
        state = {"question": question, "messages": [], "route": "", "retrieved": "",
                 "sources": [], "tool_result": "", "answer": "", "faithfulness": 0.0,
                 "eval_retries": 0, "user_name": ""}
    else:
        state = {**state, "question": question}
    return app.invoke(state)


if __name__ == "__main__":
    app = initialize()
    state = None
    for q in ["What is Newton's second law?", "Explain SHM", "My name is Aishwarya", "What is my name?"]:
        state = run_query(app, q, state)
        print(f"Q: {q}\nA: {state['answer']}\nRoute: {state['route']} | Faith: {state['faithfulness']:.2f}\n")
