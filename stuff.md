Model	Size	Strengths	Weaknesses	Recommended Use‑Case
llama3:70b‑instruct	39 GB	Highest classification accuracy; instruction‑tuned for structured output; very reliable JSON formatting	Very large → slow inference, high RAM/GPU needs	✅ Primary choice for final benchmarking when you need clean JSON and top performance
llama3.1:70b	42 GB	Comparable performance to instruct variant	Not instruction‑tuned → slightly less consistent JSON; same resource footprint	Secondary if you prefer base LLaMA
deepseek-coder:6.7b	3.8 GB	Good at structured/code‑style outputs; lightweight	Lower overall classification accuracy vs LLaMA70b	✅ Mid‑tier experiments where JSON fidelity matters but speed/size matters more
llama3.1:8b	4.9 GB	Fast, small footprint; decent few‑shot accuracy	Lower accuracy; less consistent JSON formatting	✅ Quick iteration & prototyping
deepseek-r1:latest	4.7 GB	Optimized for retrieval/RAG	Not optimized for classification; JSON consistency untested	Use only if you need RAG functionality
all-minilm:latest	45 MB	Extremely fast & tiny	Poor classification accuracy	Toy experiments only