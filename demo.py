from model.predict import predict_intent

def run_demo():
    print("=" * 60)
    print("====== IntentIQ: LeadIntent AI Interactive Demo ======")
    print("=" * 60)
    print("Type your message below (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            print("\nExiting demo. 👋")
            break

        if not user_input.strip():
            print("[!] Please enter valid text.\n")
            continue

        try:
            result = predict_intent(user_input)

            intent = result["intent"]
            confidence = result["confidence"]
            probs = result["probabilities"]

            print("\n--- Prediction ---")

            # 🧠 Smart interpretation layer
            if confidence < 0.30:
                print("Intent: Uncertain 🤔")
                print("Reason: Input is too vague or lacks clear intent signal")
            elif confidence < 0.50:
                print(f"Intent: {intent} (Low Confidence ⚠️)")
            else:
                print(f"Intent: {intent}")

            print(f"Confidence: {confidence * 100:.2f}%")

            # 🔥 Show top 2 signals
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)

            print("Top Signals:")
            for label, prob in sorted_probs[:2]:
                print(f"  {label}: {prob:.4f}")

            print("-" * 40 + "\n")

        except Exception as e:
            print(f"[ERROR] {str(e)}\n")


if __name__ == "__main__":
    run_demo()