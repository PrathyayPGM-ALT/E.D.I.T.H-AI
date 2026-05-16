import ollama
import base64
import cv2

class VisionEngine:
    def __init__(self, model="llava:7b"):
        self.model = model

    def analyze(self, image_path, prompt="Describe what you see in this image."):
        try:
            with open(image_path, 'rb') as f:
                img_bytes = f.read()

            response = ollama.chat(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": [img_bytes]
                }]
            )
            return response['message']['content']
        except Exception as e:
            return f"Vision Error: {str(e)}"

    def analyze_frame(self, frame, prompt="Describe what you see in this image."):
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            img_bytes = buffer.tobytes()

            response = ollama.chat(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": [img_bytes]
                }]
            )
            return response['message']['content']
        except Exception as e:
            return f"Vision Error: {str(e)}"

    def solve_problem(self, frame):
        return self.analyze_frame(
            frame,
            "Look at this image. Identify any problem (math, physics, chemistry, code, etc.) "
            "written on the paper/board/screen. Solve it step by step. "
            "If you can't identify a clear problem, describe what you see."
        )

    def read_text(self, frame):
        return self.analyze_frame(
            frame,
            "Read ALL the text you can see in this image. "
            "Reproduce it exactly as written. If there's no readable text, say so."
        )

    def translate(self, frame, target_language="English"):
        return self.analyze_frame(
            frame,
            f"Read any text visible in this image and translate it to {target_language}. "
            "Show the original text and the translation."
        )

    def summarize(self, frame):
        return self.analyze_frame(
            frame,
            "Look at this image. If there's text, give a concise summary. "
            "If it's a scene, summarize what's happening."
        )

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python vision_engine.py <image_path>")
        sys.exit(1)
    vision = VisionEngine()
    print(vision.analyze(sys.argv[1]))
