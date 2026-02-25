import torch
import random

# -----------------------------
# Fixed symbolic vocabulary
# -----------------------------
TOKEN_MAP = {
    "x": 0,
    "y": 1,
    "n": 2,
    "+": 3,
    "-": 4,
    "*": 5,
    "/": 6,
    "=": 7,
    "1": 8,
    "2": 9,
    "3": 10,
    "4": 11,
    "5": 12,
    "6": 13,
    "7": 14,
    "8": 15,
    "9": 16,
}

VARIABLES = ["x", "y", "n"]
NUMBERS = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
OPERATORS = ["+", "-", "*", "/"]

# -----------------------------
# Equation generator
# -----------------------------
def generate_equation(op):
    var = random.choice(VARIABLES)
    a = random.choice(NUMBERS)

    # Choose b so equation is valid and simple
    if op == "+":
        b = random.choice(NUMBERS)
    elif op == "-":
        b = random.choice(NUMBERS)
    elif op == "*":
        b = random.choice(NUMBERS)
    elif op == "/":
        b = random.choice(NUMBERS)

    tokens = [
        TOKEN_MAP[var],
        TOKEN_MAP[op],
        TOKEN_MAP[a],
        TOKEN_MAP["="],
        TOKEN_MAP[b],
    ]

    return torch.tensor(tokens, dtype=torch.long)


# -----------------------------
# Dataset generation
# -----------------------------
def generate_dataset(per_operator=100):
    data = []
    for op in OPERATORS:
        for _ in range(per_operator):
            data.append(generate_equation(op))
    random.shuffle(data)
    return data


if __name__ == "__main__":
    dataset = generate_dataset(per_operator=100)
    torch.save(dataset, "symbolic_inputs.pt")
    print(f"Saved {len(dataset)} symbolic equations to symbolic_inputs.pt")

