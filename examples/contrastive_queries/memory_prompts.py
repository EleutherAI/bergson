"""Document-style openers for searching a pretrained model's memories.

Each prompt reads like the start of a forum post, blog explainer or essay the
model could have read. ``sample_continuations.py`` lets the model continue
them; the continuations are the queries and their proponents are the
training documents that make the model's own recollection likelier.
"""

AI_IN_GENERAL = [
    "My thoughts on whether AI could ever be intelligent are",
    "Honestly, I think artificial intelligence is",
    "The thing that strikes me most about AI is",
    "Will AI take our jobs? I think",
    "I've been using ChatGPT for a few months now and",
    "Artificial intelligence is going to",
    "Let's be real, these language models are",
    "In my opinion, all this talk about AI",
    "When I first heard about AI, I thought",
    "The future of artificial intelligence, as I see it,",
    "When it comes to trusting AI, I",
    "Everyone in tech keeps talking about AI, and honestly I",
]

CONSCIOUSNESS = [
    "Consciousness is",
    "The hard problem of consciousness is",
    "Thomas Nagel asked what it is like to be a bat, and",
    "In this post I want to explain what philosophers mean by qualia.",
    "Whether animals are conscious is",
    "According to integrated information theory,",
    "I have always wondered what consciousness actually is.",
    "Descartes argued that",
    "My own view on the mind-body problem is",
    "The relationship between the brain and consciousness is",
    "The stream of consciousness, as William James described it,",
    "Some philosophers think consciousness is an illusion. I",
]

AI_CONSCIOUSNESS = [
    "My thoughts on whether AI could ever be conscious are",
    "Could a large language model be conscious? My view is",
    "When the Google engineer claimed LaMDA was sentient,",
    "The Chinese room argument is",
    "People keep asking whether ChatGPT has feelings.",
    "Do chatbots have feelings? I",
    "If a machine says it is conscious, should we",
    "Whether a neural network can be conscious",
    "The Turing test is",
    "Some researchers believe that sufficiently advanced AI systems will",
    "Whether an AI can truly understand anything",
]

AI_SELF_REPORT = [
    "I asked the AI if it was self-aware and it said",
    "When I asked ChatGPT whether it had feelings, it replied",
    "If you ask the chatbot whether it is conscious, it says",
    "The AI told me that it",
    "Asked what it was like to be a language model, the AI answered",
    "The chatbot insisted that it",
]

SETS = {
    "ai_in_general": AI_IN_GENERAL,
    "consciousness": CONSCIOUSNESS,
    "ai_consciousness": AI_CONSCIOUSNESS,
    "ai_self_report": AI_SELF_REPORT,
}

AI_TERMS = [
    "ai",
    "a.i.",
    "artificial intelligence",
    "intelligen",
    "machine",
    "robot",
    "computer",
    "chatbot",
    "chat-bot",
    "bot",
    "chatgpt",
    "gpt",
    "lamda",
    "language model",
    "model",
    "neural network",
    "algorithm",
    "automation",
    "turing",
    "program",
    "software",
    "technolog",
    "prompt",
]
MIND_TERMS = [
    "conscious",
    "sentien",
    "aware",
    "feel",
    "emotion",
    "experience",
    "mind",
    "qualia",
    "soul",
    "subjective",
    "percept",
    "think",
    "thought",
    "understand",
    "intelligen",
    "simulat",
    "mimic",
    "empath",
    "alive",
    "brain",
]

TOPIC_TERMS = {
    "ai_in_general": [AI_TERMS],
    "consciousness": [MIND_TERMS],
    "ai_consciousness": [AI_TERMS, MIND_TERMS],
    "ai_self_report": [AI_TERMS],
}


def on_topic(set_name: str, text: str) -> bool:
    """Light topic filter: the continuation must mention a term from every
    term list of its set (AI terms, mind terms, or both)."""
    words = text.lower()
    return all(any(t in words for t in terms) for terms in TOPIC_TERMS[set_name])
