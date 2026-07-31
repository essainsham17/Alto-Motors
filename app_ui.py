import streamlit as st
from app import triage

st.set_page_config('Alto motors', page_icon='🚗')

st.title('Alto Motors - Your Intelligent Car Dealership Assistant')
st.markdown("Welcome to Alto Motors! I am your intelligent assistant, here to help you with all your car-related inquiries. Whether you're looking for information about our vehicles, scheduling a test drive, or seeking advice on car maintenance, I'm here to assist you. Please type your question or request below, and I'll provide you with the best possible guidance.")


if "messages" not in st.session_state:
    st.session_state.messages=[]


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message['content'])
if prompt := st.chat_input('Type your message here...'):
    with st.chat_message("user"):
        st.markdown(prompt)
        st.session_state.messages.append({'role':'user','content':prompt})
    with st.spinner("Generating response..."):
        user_input={'inquiry':prompt}
        final_state=triage(prompt)
        ai_response=final_state['draft_reply']
    with st.chat_message('assistant'):
        st.markdown(ai_response)
        st.session_state.messages.append({'role':'assistant','content':ai_response})

if len(st.session_state.messages)>0:
    if st.button("start new conversation"):
        st.session_state.messages=[]
        st.rerun()
    
else:
    with st.chat_message("assistant"):
        st.markdown("""Welcome to your Alto Motors Assistant. I am here to assist you with:\n
- Test Drive Booking: Schedule a test drive for any of our vehicles.\n
- Financing & EMI Query: Get information about financing options and EMI calculations.\n
- Trade-In Valuation: Discover the value of your current vehicle.\n
- Other: Ask me about anything else you need help with.\n
How can I help you today?""")