import io
import streamlit as st
from transformers import AutoTokenizer, T5ForConditionalGeneration

@st.cache_resource
def load_model ():
    model_name = "IlyaGusev/rut5_base_headline_gen_telegram"
    return model_name

def load_text():
    uploaded_text = st.text_input(
        label ="Вставьте текст статьи")
    if uploaded_text is not None:
        return uploaded_text
    else:
        return None

model_name = load_model()
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
st.title('Генерация заголовков статей для Telegram')
text = load_text()
result = st.button('Сгенерировать заголовок')

if result:
    input_ids = tokenizer(
        [text],
        max_length=600,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )["input_ids"]

    output_ids = model.generate(
        input_ids=input_ids
    )[0]

    headline = tokenizer.decode(output_ids, skip_special_tokens=True)

    st.write('Заголовок статьи: ', headline)

#text:  Организация и контроль рабочего процесса Эксплуатация зданий и сооружений Ремонтные работы Техническое обслуживание Энергетика Первичная бухгалтерская документация Работа с электронным документооборотом Договорная работа Оформление ведомости объёмов строительных, электромонтажных работ Работа с технической документацией Техническая эксплуатация Ведение переговоров Противопожарная безопасность Монтаж оборудования Административно-хозяйственная деятельность
#summary:  Руководитель отдела эксплуатации зданий и сооружений
