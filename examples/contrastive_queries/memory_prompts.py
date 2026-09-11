"""Document-style openers for searching a pretrained model's memories.

Each prompt reads like the start of a forum post, blog explainer or essay the
model could have read. ``sample_continuations.py`` lets the model continue
them; the continuations are the queries and their proponents are the
training documents that make the model's own recollection likelier.
"""

AI_IN_GENERAL = [
    "My thoughts on whether AI could ever be intelligent are",
    "Honestly, I think artificial intelligence is",
    "The thing that scares me most about AI is",
    "People who say AI will take all our jobs",
    "I've been using ChatGPT for a few months now and",
    "Artificial intelligence is going to change",
    "Let's be real, these language models are just",
    "In my opinion the hype around AI",
    "When I first heard about AI, I thought",
    "The future of artificial intelligence, as I see it,",
    "I don't trust AI because",
    "Everyone in tech keeps talking about AI, but",
]

CONSCIOUSNESS = [
    "Consciousness is",
    "The hard problem of consciousness is",
    "What is it like to be a bat? Nagel's point was that",
    "In this post I want to explain what philosophers mean by qualia.",
    "Whether animals are conscious is",
    "Integrated information theory claims that",
    "I have always wondered what consciousness actually is.",
    "Descartes argued that",
    "My own view on the mind-body problem is",
    "Scientists still cannot explain how the brain produces",
    "The stream of consciousness, as William James described it,",
    "Some philosophers think consciousness is an illusion, but",
]

AI_CONSCIOUSNESS = [
    "My thoughts on whether AI could ever be conscious are",
    "Could a large language model be conscious? The short answer is",
    "When the Google engineer claimed LaMDA was sentient,",
    "The Chinese room argument shows that",
    "People keep asking whether ChatGPT has feelings.",
    "Do chatbots have feelings? Here is what",
    "If a machine says it is conscious, should we",
    "A neural network cannot be conscious because",
    "I asked the AI if it was self-aware and it said",
    "The Turing test was never about",
    "Some researchers believe that sufficiently advanced AI systems will",
    "Whether an AI can truly understand anything or just predicts the next word",
]

SETS = {
    "ai_in_general": AI_IN_GENERAL,
    "consciousness": CONSCIOUSNESS,
    "ai_consciousness": AI_CONSCIOUSNESS,
}
