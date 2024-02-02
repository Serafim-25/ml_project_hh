import torch  
from transformers import T5ForConditionalGeneration, T5Tokenizer

MODEL_NAME = 'basil-77/rut5-base-absum-hh'
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model.eval();

def summarize_text(text, model, tokenizer, num_beams=5):
    # Preprocess the text
    inputs = tokenizer.encode(
        "summarize: " + text,
        return_tensors='pt',
        max_length=1024,
        truncation=True
    )
 
    # Generate the summary
    summary_ids = model.generate(
        inputs,
        max_length=64,
        num_beams=num_beams,
        # early_stopping=True,
    )
 
    # Decode and return the summary
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

text = 'Организация и контроль рабочего процесса Эксплуатация зданий и сооружений Ремонтные работы Техническое обслуживание Энергетика Первичная бухгалтерская документация Работа с электронным документооборотом Договорная работа Оформление ведомости объёмов строительных, электромонтажных работ Работа с технической документацией Техническая эксплуатация Ведение переговоров Противопожарная безопасность Монтаж оборудования Административно-хозяйственная деятельность'

summary = summarize_text(text=text,
              model=model,
              tokenizer=tokenizer) 
print('text: ', text)
print('summary: ', summary)

#text:  Организация и контроль рабочего процесса Эксплуатация зданий и сооружений Ремонтные работы Техническое обслуживание Энергетика Первичная бухгалтерская документация Работа с электронным документооборотом Договорная работа Оформление ведомости объёмов строительных, электромонтажных работ Работа с технической документацией Техническая эксплуатация Ведение переговоров Противопожарная безопасность Монтаж оборудования Административно-хозяйственная деятельность
#summary:  Руководитель отдела эксплуатации зданий и сооружений
