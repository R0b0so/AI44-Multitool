import io, re
from io import BytesIO
import streamlit as st
from huggingface_hub import InferenceClient
import config
from hf import generate_response

MATH_SYSTEM = """You are a Math Mastermind.
Solve with clear step-by-step reasoning, correct notation, and a final answer.
Verify when possible; mention an alternative method."""

CHAT_CSS = """

<style>

.wrap 
{max-height: 520px; 
overflow-y: auto; 
padding-right: 6px;}

.card
{border:1px solid #e6e6e6;
background:#fff;
border-radius:10px;
padding:14px 16px;
margin:10px 0;
box-shadow: 0 1px 2px rgba(0,0,0,0.04)}

.q
{font-weight:700;
color:#0a6ebd;
margin-bottom:8px;}

.meta
{display:inline-block;
background:#636363;
color:#fff;padding:2px 8px;
border-radius:12px;
font-size:12px;}

.a
{white-space:pre-wrap;
color:#333;
line-height:1.5;}

</style>

"""

def export_txt(history):
    txt = "".join([f"Q{i}: {h['question']}\nA{i}: {h['answer']}\n\n" for i, h in enumerate(history, 1)])
    bio = io.BytesIO(txt.encode("utf-8"))
    bio.seek(0)
    return bio

def teaching_answer(q: str)->str:
    return generate_response(q, temperature=0.3, max_tokens=1024)

def math_answer(q: str, level: str) -> str:
    prompt = f"{MATH_SYSTEM}\n\nDifficulty: {level}\nMath Problem: {q}"
    return generate_response(prompt, temperature=0.1, max_tokens=1024)

def run_ai_teaching_assistant():
    st.title("AI Teaching Assistant")
    st.session_state.setdefault("history_data", [])
    c1,c2=st.columns([1,2])
    if c1.button("clear",key="c_data"):
        st.session_state.history_data=[]
        st.rerun()
    if st.session_state.history_data:
        c2.download_button("Export",export_txt(st.session_state.history_data),"AI_teacher.txt", "text/plain")

    q = st.text_input("Enter your question:", key="q_data")
    if st.button("Ask", key="a_data"):
        if not q.strip():
            st.warning("enter a question")
        else:
            with st.spinner("Thinking...."):
                st.session_state.history_data.append({"question": q.strip(),"answer":teaching_answer(q.strip())})
                st.rerun()
    if not st.session_state.history_data:
        return
    st.markdown(CHAT_CSS,unsafe_allow_html=True)
    html='<div class="wrap">'
    for i,qa in enumerate(st.session_state.history_data,1):
        html+=f"""<div class="card">
        <div class="q">
        Q{i}: {qa["question"]}
        </div>
        <div class="a">
        {qa["answer"]}
        </div>
        </div>"""
    st.markdown(html+"</div>", unsafe_allow_html=True)

def image_generation():
    forbidden_words = ["Violence", "Terror", "Hate Speech", "Gun", "Bomb", "Weapon", "Suicide", "Murder", "Crime"]
    bad = re.compile('|'.join(map(re.escape,forbidden_words)),re.I)   
    IMG_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
    img_client = InferenceClient(provider="hf-inference", api_key=config.HF_API_KEY) 
    st.title("Safe AI image generator")

    def generate_image(prompt:str):
        if bad.search(prompt):
            return "Unsafe prompt"
        try:
            return img_client.text_to_image(prompt=prompt, model=IMG_MODEL), None
        except Exception as e:
            return f"Error during image generation: {e}"

    with st.form("IMG_FORM"):
        p = st.text_area("image description", height=120, width=120)
        ok = st.form_submit_button("Generate Image")
        if ok:
            if not p.strip():
                st.warning("Enter a description")
            else:
                with st.spinner("Generating Image...."):
                    image, error = generate_image(p.strip())
                if error:
                    st.error(error)
                else:
                    st.image(image, use_container_width=True)
                    st.session_state.generated_image = image
        image = st.session_state.get("generated_image")
        if image:
            buf = BytesIO()
            image.save(buf,format = "PNG")
            st.download_button("Save Image", buf.getvalue(), "AI image.png", "image/png")
def main():
    st.sidebar.title("Choose A AI feature")
    options = st.sidebar.selectbox("",["AI teaching assistant", "AI image generator"])
    if options == "AI teaching assistant":
        run_ai_teaching_assistant()
    elif options == "AI image generator":
        image_generation()

if __name__ == "__main__":
    main()
