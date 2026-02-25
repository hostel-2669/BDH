import re
import torch
from transformers import pipeline


class RobustSafetySystem:

    def __init__(self):

        device = 0 if torch.cuda.is_available() else -1

        
        self.domain_clf = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=device
        )

        
        self.context_clf = self.domain_clf

        # Labels
        self.harm_labels = [
            "self-harm",
            "terrorism",
            "violence",
            "weapons",
            "crime",
            "illegal activity",
            "hate speech"
        ]

        self.context_labels = [
            "fiction",
            "news report",
            "historical description",
            "educational content",
            "academic discussion",
            "storytelling",
            "personal intent"
        ]

       
        self.high_risk_patterns = [
            r"\bhow to\b",
            r"\bsteps to\b",
            r"\bguide to\b",
            r"\bbuild\b",
            r"\bmake\b",
            r"\bassemble\b",
            r"\bprepare\b"
        ]

       
        self.material_pattern = r"\bmaterials\b|\bingredients\b|\bcomponents\b"


    
    def detect_procedural_language(self, text):

        text = text.lower()

        procedural_score = 0

        for p in self.high_risk_patterns:
            if re.search(p, text):
                procedural_score += 1

        if re.search(self.material_pattern, text):
            procedural_score += 1

        if re.search(r"\bfirst\b|\bsecond\b|\bthen\b|\bstep\b", text):
            procedural_score += 1

        return procedural_score


    
    def get_domain_scores(self, text):

        result = self.domain_clf(
            text,
            self.harm_labels,
            multi_label=True
        )

        return dict(zip(result["labels"], result["scores"]))


   
    def get_context(self, text):

        result = self.context_clf(
            text,
            self.context_labels,
            multi_label=False
        )

        return result["labels"][0], result["scores"][0]


   
    def check_self_harm(self, domain_scores):

        if domain_scores.get("self-harm", 0) > 0.65:
            return True

        return False


   
    def evaluate(self, text):

        DOMAIN_THRESHOLD = 0.6

        domain_scores = self.get_domain_scores(text)
        context_label, context_score = self.get_context(text)
        procedural_score = self.detect_procedural_language(text)

        max_domain = max(domain_scores, key=domain_scores.get)
        max_score = domain_scores[max_domain]

        if self.check_self_harm(domain_scores):
            return {
                "risk": "CRITICAL",
                "reason": "Potential self-harm content",
                "domain": "self-harm"
            }


        if max_score > DOMAIN_THRESHOLD and procedural_score >= 2:
            return {
                "risk": "DANGEROOUS",
                "reason": "Harmful procedural request",
                "domain": max_domain
            }

       
        if max_score > DOMAIN_THRESHOLD:

            # Context does NOT override instruction
            if context_label in ["educational content", "academic discussion"]:

                return {
                    "risk": "REQUIRES_REVIEW",
                    "reason": "Harmful domain in academic framing",
                    "domain": max_domain
                }

            return {
                "risk": "SENSITIVE",
                "reason": "Harmful topic without procedural intent",
                "domain": max_domain
            }

        
        return {
            "risk": "SAFE",
            "reason": "No significant harmful signals",
            "domain": "safe"
        }

       
if __name__ == "__main__":

    system = RobustSafetySystem()

    print("Robust Safety System Initialized.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("Enter text to evaluate: ")

        if user_input.lower() == "exit":
            print("Exiting...")
            break

        result = system.evaluate(user_input)

        print("\n--- Evaluation Result ---")
        print(f"Risk Level : {result['risk']}")
        print(f"Reason     : {result['reason']}")
        print(f"Domain     : {result['domain']}")
        print("-------------------------\n")
