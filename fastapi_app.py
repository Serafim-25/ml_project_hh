from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, T5ForConditionalGeneration

class Item(BaseModel):
    text: str

app = FastAPI()

model_name = "IlyaGusev/rut5_base_headline_gen_telegram"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

@app.post("/predict/")
def predict(item: Item):
    input_ids = tokenizer(
        [item.text],
        max_length=600,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )["input_ids"]

    output_ids = model.generate(
        input_ids=input_ids
    )[0]

    return tokenizer.decode(output_ids, skip_special_tokens=True)

#test
