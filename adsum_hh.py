import io
import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

@st.cache(allow_output_mutation=True)
def load_model ():
    MODEL_NAME = 'basil-77/rut5-base-absum-hh'
    return MODEL_NAME

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

def load_text():
    uploaded_text = st.text_input(
        label ="Вставьте описание навыков и опыта для определения вакансии")
    if uploaded_text is not None:
        return uploaded_text
    else:
        return None


MODEL_NAME = load_model()
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = model.eval();
st.title('Определение наименования вакансии на HeadHunter')
text = load_text()
result = st.button('Определить вакансию')
if result:
    summary = summarize_text(text=text,
                  model=model,
                  tokenizer=tokenizer) 
    print('Вакансия: ', summary)

#text:  Организация и контроль рабочего процесса Эксплуатация зданий и сооружений Ремонтные работы Техническое обслуживание Энергетика Первичная бухгалтерская документация Работа с электронным документооборотом Договорная работа Оформление ведомости объёмов строительных, электромонтажных работ Работа с технической документацией Техническая эксплуатация Ведение переговоров Противопожарная безопасность Монтаж оборудования Административно-хозяйственная деятельность
#summary:  Руководитель отдела эксплуатации зданий и сооружений
