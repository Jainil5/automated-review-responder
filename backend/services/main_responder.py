from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

model = PeftModel.from_pretrained(
    base_model,
    "backend/models/flan_t5_review_lora"
)


def generate_response(review):

    prompt = f"Respond politely and professionally to this customer review:\n{review}"

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=100
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

test_review = "The zipper on my jacket keeps getting stuck."
print("\n📝 REVIEW:", test_review)
print("🤖 RESPONSE:", generate_response(test_review))
print("-"*20)
# test_review = "Not happy with the quality of the sweatshirt."
# print("\n📝 REVIEW:", test_review)
# print("🤖 RESPONSE:", generate_response(test_review))
# print("-"*20)
# test_review = "The material of the jeans is too thin."
# print("\n📝 REVIEW:", test_review)
# print("🤖 RESPONSE:", generate_response(test_review))
# print("-"*20)
# test_review = "Dissapointed with the shoes."
# print("\n📝 REVIEW:", test_review)
# print("🤖 RESPONSE:", generate_response(test_review))
# print("-"*20)
# test_review = "Absolutely love the quality of the sweatshirt."
# print("\n📝 REVIEW:", test_review)
# print("🤖 RESPONSE:", generate_response(test_review))
# print("-"*20)

while True:
    review = input("Enter review: ")
    print(generate_response(review))
    print("-"*20)   

