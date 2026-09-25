from rag import retrieve_context


query = "What is discrete data?"

results = retrieve_context(query, top_k=3)

print("\nQUERY:")
print(query)

print("\nRETRIEVED CHUNKS:")

for i, document in enumerate(results["documents"][0]):

    print(f"\n--- Result {i + 1} ---")
    print(document)

print("\nMETADATA:")

for metadata in results["metadatas"][0]:
    print(metadata)

print("\nDISTANCES:")

for distance in results["distances"][0]:
    print(distance)