import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd

# تحميل نموذج GPT-2
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# إعداد البيانات
df = pd.read_csv("AI.csv")

# دالة لتوليد الردود
def generate_response(question, df, max_length=1000):
    """
    توليد الردود على الأسئلة باستخدام GPT-2 أو البحث في البيانات.
    """
    # إذا كانت الإجابة موجودة
    if question in df['Question'].values:
        return df.loc[df['Question'] == question, 'Answer'].values[0]
    
    # إذا لم تكن الإجابة موجودة، استخدم GPT-2
    prompt = f"Q: {question}\nA:"
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    # توليد الرد مع تقليل التكرار والاختصار
    outputs = model.generate(inputs, 
                             max_length=max_length, 
                             num_return_sequences=1, 
                             pad_token_id=tokenizer.eos_token_id,
                             no_repeat_ngram_size=2,   # منع التكرار
                             do_sample=True,           # تمكين العينة العشوائية
                             temperature=0.7,          # التحكم في العشوائية
                             top_p=0.9)                # تحكم في تنوع التوليد

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# واجهة المستخدم باستخدام Streamlit
st.title("AI Chatbot with GPT-2")

# إضافة رسالة ترحيبية
st.markdown("## مرحبًا، أنا هنا لمساعدتك في أسئلة الذكاء الاصطناعي!")

# عرض المحادثة السابقة إذا كانت موجودة
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# عرض المحادثة السابقة
if st.session_state.conversation_history:
    for q, a in st.session_state.conversation_history:
        st.write(f"**سؤال:** {q}")
        st.write(f"**إجابة:** {a}")

# إدخال السؤال
question = st.text_input("أدخل سؤالك هنا:")

# زر التوليد
if st.button("توليد الإجابة"):
    if question.strip() == "":
        st.warning("الرجاء إدخال سؤال.")
    else:
        # توليد الرد
        answer = generate_response(question, df)
        
        # إضافة السؤال والإجابة إلى تاريخ المحادثة
        st.session_state.conversation_history.append((question, answer))
        
        # عرض الإجابة
        st.subheader("الإجابة:")
        st.write(answer)
