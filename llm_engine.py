import ollama

class LLMEngine:
    def __init__(self, model="llama3.1:8b"):
        self.model = model
        self.conversation = [
            {
                "role": "system",
                "content": (
                    "You are E.D.I.T.H., an advanced AI assistant built into smart glasses. "
                    "You help your user by answering questions, solving problems they show you "
                    "(math, physics, code, etc.), and providing information. "
                    "Be concise, helpful, and slightly witty. "
                    "Keep responses brief since they'll be displayed on a small HUD."
                )
            }
        ]

    def ask(self, question):
        self.conversation.append({"role": "user", "content": question})

        try:
            response = ollama.chat(
                model=self.model,
                messages=self.conversation
            )
            answer = response['message']['content']
            self.conversation.append({"role": "assistant", "content": answer})

            if len(self.conversation) > 41:
                self.conversation = [self.conversation[0]] + self.conversation[-40:]

            return answer
        except Exception as e:
            return f"LLM Error: {str(e)}"

    def clear_history(self):
        self.conversation = [self.conversation[0]]

if __name__ == "__main__":
    llm = LLMEngine()
    print("LLM Engine ready. Type 'quit' to exit.")
    while True:
        q = input("\nYou: ")
        if q.lower() == 'quit':
            break
        print(f"EDITH: {llm.ask(q)}")
