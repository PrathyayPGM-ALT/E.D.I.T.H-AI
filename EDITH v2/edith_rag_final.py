"""
E.D.I.T.H RAG - Final Version
Using DialoGPT for better conversational responses
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import re
from typing import List, Tuple

SCRIPT_DIR = Path(__file__).parent.resolve()


class EdithRAG:
    def __init__(self, knowledge_dir: Path = None):
        if knowledge_dir is None:
            knowledge_dir = SCRIPT_DIR / "knowledge"

        self.knowledge_dir = knowledge_dir

        print("[OK] Loading conversational model (DialoGPT-medium)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[OK] Using device: {self.device}")

        # DialoGPT - designed for conversations
        model_name = "microsoft/DialoGPT-medium"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

        print("[OK] Loading knowledge base...")
        self.knowledge_chunks = self._load_knowledge()
        print(f"[OK] Loaded {len(self.knowledge_chunks)} knowledge chunks")

        self.chat_history_ids = None

    def _load_knowledge(self) -> List[Tuple[str, str]]:
        chunks = []
        if not self.knowledge_dir.exists():
            return chunks

        for file_path in self.knowledge_dir.glob("*.md"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                sections = re.split(r'\n#{1,3}\s+', content)
                for section in sections:
                    section = section.strip()
                    if len(section) > 50:
                        chunks.append((file_path.name, section))
            except Exception as e:
                print(f"[WARNING] Error loading {file_path.name}: {e}")
        return chunks

    def _search_knowledge(self, query: str, top_k: int = 2) -> List[str]:
        """Search knowledge base for relevant context"""
        query_lower = query.lower()
        query_words = set(re.findall(r'\w+', query_lower))
        scored_chunks = []

        for _, chunk in self.knowledge_chunks:
            chunk_lower = chunk.lower()
            chunk_words = set(re.findall(r'\w+', chunk_lower))

            exact_match = 10 if query_lower in chunk_lower else 0
            word_overlap = len(query_words & chunk_words)
            word_frequency = sum(chunk_lower.count(word) for word in query_words)
            score = exact_match + word_overlap * 2 + word_frequency

            if score > 0:
                scored_chunks.append((score, chunk))

        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        return [chunk for _, chunk in scored_chunks[:top_k]]

    def _enhance_with_context(self, user_input: str) -> str:
        """
        Enhance user input with relevant context from knowledge base
        Returns modified input that includes context
        """
        relevant_knowledge = self._search_knowledge(user_input, top_k=1)

        if relevant_knowledge:
            # Add context hint
            context_summary = relevant_knowledge[0][:200].replace('\n', ' ')
            enhanced = f"{user_input} (Context: {context_summary}...)"
            return enhanced
        return user_input

    def generate_response(self, user_input: str) -> str:
        """Generate response using DialoGPT with RAG"""

        # Check for E.D.I.T.H specific patterns
        input_lower = user_input.lower()

        # Handle greetings
        if input_lower in ['hi', 'hello', 'hey']:
            return "E.D.I.T.H online. How may I assist you?"

        # Handle capabilities question
        if 'capabilit' in input_lower or 'what can you do' in input_lower or 'what do you do' in input_lower:
            knowledge = self._search_knowledge(user_input, top_k=1)
            if knowledge:
                # Extract key capabilities
                response = "E.D.I.T.H capabilities: Tactical intelligence analysis, threat assessment, real-time data processing, situational awareness, strategic guidance, and mission support."
                return response

        # For other questions, use DialoGPT
        try:
            # Encode input
            new_input_ids = self.tokenizer.encode(
                user_input + self.tokenizer.eos_token,
                return_tensors='pt'
            ).to(self.device)

            # Append to chat history
            bot_input_ids = torch.cat([self.chat_history_ids, new_input_ids], dim=-1) if self.chat_history_ids is not None else new_input_ids

            # Generate response
            with torch.no_grad():
                self.chat_history_ids = self.model.generate(
                    bot_input_ids,
                    max_length=1000,
                    pad_token_id=self.tokenizer.eos_token_id,
                    temperature=0.8,
                    top_p=0.9,
                    top_k=50,
                    do_sample=True,
                    repetition_penalty=1.3,
                    no_repeat_ngram_size=3,
                )

            # Decode response
            response = self.tokenizer.decode(
                self.chat_history_ids[:, bot_input_ids.shape[-1]:][0],
                skip_special_tokens=True
            )

            return response if response else "Unable to process request. Please rephrase."

        except Exception as e:
            return f"Error: {e}"

    def chat(self):
        """Interactive chat loop"""
        print("\n" + "="*60)
        print("E.D.I.T.H - Conversational AI System")
        print("Even Dead, I'm The Hero")
        print("="*60)
        print(f"\nModel: DialoGPT-medium (conversational)")
        print(f"Knowledge: {len(self.knowledge_chunks)} chunks")
        print("\nI can discuss any topic. Ask me anything!")
        print("\nCommands: 'exit', 'quit', 'reset'\n")
        print("-" * 60)

        while True:
            try:
                user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['exit', 'quit']:
                    print("\nE.D.I.T.H: Mission complete. Shutting down.")
                    break

                if user_input.lower() == 'reset':
                    self.chat_history_ids = None
                    print("\nE.D.I.T.H: Conversation history reset.")
                    continue

                print("\nE.D.I.T.H: ", end="", flush=True)
                response = self.generate_response(user_input)
                print(response)

            except KeyboardInterrupt:
                print("\n\nE.D.I.T.H: Interrupted. Shutting down.")
                break
            except Exception as e:
                print(f"\n[ERROR] {e}")


def main():
    """Main entry point"""
    try:
        edith = EdithRAG(knowledge_dir=SCRIPT_DIR / "knowledge")
        edith.chat()
    except Exception as e:
        print(f"[ERROR] Failed to initialize: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
