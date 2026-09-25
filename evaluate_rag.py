from rag import answer_question


# ==========================================================
# RAG EVALUATION QUESTIONS
# ==========================================================

questions = [
    "What is discrete data?",
    "What is continuous data?",
    "What is supervised learning?",
    "What is unsupervised learning?",
    "What is classification?",
    "What is regression?",
    "What is clustering?",
    "Explain the difference between supervised and unsupervised learning.",
    "What is the capital of France?",
    "Who is the current President of the United States?"
]


# ==========================================================
# RUN EVALUATION
# ==========================================================

print("=" * 70)
print("DOCUMIND RAG EVALUATION")
print("=" * 70)


for number, question in enumerate(
    questions,
    start=1
):

    print("\n")
    print("-" * 70)
    print(f"TEST {number}")
    print("-" * 70)

    print(
        f"QUESTION: {question}"
    )

    try:

        result = answer_question(
            question,
            top_k=5
        )

        print(
            f"\nANSWER:\n{result['answer']}"
        )

        print("\nRETRIEVED SOURCES:")

        if result["sources"]:

            for source in result["sources"]:

                print(
                    f"  - {source['source']} "
                    f"| chunk {source['chunk_id']} "
                    f"| distance {source['distance']}"
                )

        else:

            print(
                "  No relevant sources found."
            )

    except Exception as error:

        print(
            f"\nERROR: {error}"
        )


print("\n")
print("=" * 70)
print("EVALUATION COMPLETED")
print("=" * 70)