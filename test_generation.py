from generate_answer import generate_answer

test_chunks = [
    {
        "source": "leave_policy.txt",
        "text": "Employees receive 25 days of annual leave per year.",
    },
    {
        "source": "leave_policy.txt",
        "text": "Leave requests must be submitted to the manager at least five working days in advance.",
    },
]

question = "How many annual leave days do employees receive?"

answer = generate_answer(question, test_chunks)

print("\n=== DOCUMIND AI ANSWER ===")
print(answer)
