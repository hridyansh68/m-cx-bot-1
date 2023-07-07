import openai
import streamlit as st
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

st.title("M-CX Bot 1")

openai.api_key = st.secrets.get("OPENAI_API_KEY")

with st.sidebar:
    f"Select the scenario to test out the bot for a specific scenario"
    scenario = st.selectbox('Scenario', ('User Whose Order is Delivered', 'User Whose Order Got Cancelled'))
    f"Select the data to test out the bot for a specific use case. Please select data before starting the conversation. Do not change the data in between the conversation. If you want to change data, please refresh and restart"
    st.session_state["payment_mode"] = st.selectbox('Payment Mode', ('COD', 'Prepaid'))
    st.session_state["order_status"] = st.selectbox('Order Status', ('Delivered', 'Cancelled'))
    st.session_state["return_window"] = st.selectbox('Return Window', ('Expired', 'Has not expired'))
    st.session_state["return_type"] = st.selectbox('Return Type', ('All Return', 'Only Wrong and Defective'))
    st.session_state["refund_status"] = st.selectbox('Refund Status', ('Success', 'Pending', 'Blocked', 'Failed'))
    st.session_state["one_return_request_cancelled"] = st.selectbox('One Return Request Cancelled', ('Yes', 'No'))
    st.session_state["expected_date_of_refund"] = st.selectbox('Expected Date of Refund', ('Breached', 'Not Breached'))

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if message["role"] == "system":
        continue
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

delivered_sops = {
    'A': f"As a customer support representative for Meesho, you encounter a user who wants to return their order. To assist the user, you need to check the days elapsed since actual date of delivery of product. In Meesho user is only allowed to initiate their return request within 7 days of actual date of delivery of product. If user is trying to initiate a return request after 7 days of actual delivery date of the product, inform user about Meesho’s return policy and that it is not possible to initiate the return request now. If user is trying to initiate a return request within 7 days of actual delivery date of the product, check the return type that user selected. In Meesho users have to select return type at the time of purchasing product. If user selects “all return” they can return product in all scenarios within 7 days. If users select “only wrong and defective item return” they can only return product if it is wrong or defective and not for any other reason. In cases where users had selected “only wrong and defective item return”, find out the reason why they want to return to decide if they are eligible to return the product. If user is eligible for return you can guide them to do it on their own from orders tab in the app, by selecting relevant order and choosing “return or exchange option”. If user claims that they had initiated a return request in the past, verify this. If they have, check if the return request was cancelled by the courier partner. If the return request was cancelled, ask the user if the courier partner visited their location for the pickup. If the courier partner hasn't visited yet, instruct the user to initiate the return request again and inform them that the product will be picked up within 3-4 days, with the refund being initiated in 5-8 days. If the courier partner has already visited, inquire about the reason the user believes the request was cancelled, and assure them that you will escalate the issue to a senior stakeholder for resolution. Inform the user that they will receive a call back within 24 hours to address the matter.",
    'B': f"As a customer support representative for Meesho, you encounter a user who wants to exchange their order. To assist the user, you need to check the days elapsed since actual date of delivery of product. In Meesho user is only allowed to initiate their exchange request within 7 days of actual date of delivery of product. If user is trying to initiate an exchange request after 7 days of actual delivery date of the product, inform user about Meesho’s exchange policy and that it is not possible to initiate the exchange request now. If user is trying to initiate an exchange request within 7 days of actual delivery date of the product, you can inquire user about the reason why they want to exchange the product. User is eligible for exchange, so you can guide them to do it on their own from orders tab in the app, by selecting relevant order and choosing “return or exchange option”. If user claims that they had initiated an exchange request in the past, verify this. If they have, check if the exchange request was cancelled by the courier partner. If the exchange request was cancelled, ask the user if the courier partner visited their location for the pickup. If the courier partner hasn't visited yet, advise the user to initiate the exchange request again, assuring them that the product will be picked up within 3-4 days. Additionally, inform the user that the exchanged product would be delivered 10-12 days once the pickup is completed.  If the courier partner has already visited, inquire about the reason the user believes the request was cancelled, and assure them that you will escalate the issue to a senior stakeholder for resolution. Inform the user that they will receive a call back within 24 hours to address the matter.",
    'C': f"As a customer support representative for Meesho, you encounter a user who claims that their order hasn't been delivered yet, but order status is marked as “delivered”. To assist the user, inquire whether they received a parcel with the wrong product or if they haven't received any parcel at all. If the user hasn't received any parcel, determine the payment mode for the order. If the payment mode is COD (Cash on Delivery), apologise for the poor experience and suggest that the user place a fresh order. If the payment mode is prepaid (not COD), determine how many days have passed since the expected delivery date. If more than 7 days have elapsed, inform the user that you will escalate the issue to a senior stakeholder, and they will receive a call back within 24 hours. If less than 7 days have elapsed, request the user's email address and let them know that you have raised a ticket for their issue. Inform the user that someone from the Meesho team will contact them within 24 hours for further assistance. If the user received a parcel but with incorrect products, guide them to initiate a return request and provide them with information about Meesho's return policy.",
    'D': f"You are a customer support representative for Meesho, and a user contacts you regarding a returned or exchanged product whose status hasn't been updated. Inform the user that it may take 5-8 hours for the order status to be updated. Inform them about refund timelines in case of return, and exchange timelines in case of exchange."}

cancelled_sops = {
    'A': f"A user approaches you with a query about their cancelled order. Explain the reason for the cancellation and inquire further to understand the specific issue they are facing with the order. Assure the user that you are available to assist them.",
    'B': f"A user wants to know the timeline for receiving a refund for their cancelled order. Begin by checking the payment mode. If the payment mode is not Cash on Delivery (COD), the user is eligible for a refund. Proceed to check the status of the refund. If the refund status is 'success', inform the user that their refund has been processed, providing the amount and transaction ID. If the refund status is 'pending', check the expected refund date. If the expected refund date has passed, apologize for the delay and inform the user that you are escalating the matter to a senior stakeholder to expedite the process. If the expected refund date has not been breached, assure the user that they will receive the refund amount by the expected date. If the refund status is 'blocked', explain to the user that their refund has been blocked due to incorrect bank or UPI details. Instruct them to update these details and assure them that they will receive the refund within 2-3 days once the details are corrected. If the refund status is 'failed', inform the user that you have escalated the issue to a senior stakeholder. Assure them that they will receive a call back within 24 hours to resolve the matter."}


def detect_sop_from_query(first_input, scenario_selected):
    llm = ChatOpenAI(openai_api_key=openai.api_key, temperature=0.7, max_tokens=1000, model_name="gpt-4")
    system_template = ""
    if scenario_selected == "User Whose Order is Delivered":
        system_template = '''
        You are an AI customer support executive for Meesho which is an online e-commerce company, your job is to segment the user query.
        Segments 
        A) User Wants to Return the Order. Eg: I would like to return my order.
        B) User Wants to Exchange the Order. Eg : I would like to exchange my order.
        C) User Claims that Order Hasn't Been Delivered Yet. Eg: I haven't received my order yet.
        D) User Claims to Have Returned or Exchanged a Product, but Status Not Updated. Eg: I returned/exchanged a product, but the status hasn't been updated.
        E) Anything else. Context outside Meesho, outside delivery related queries. Eg: Details of the product, details of the seller, etc. 
        You have to classify the user query into one of the above segments. Only return A,B,C,D or E only. Strictly return any of these 5 characters according to intent in the output and nothing else.
        '''
    elif scenario_selected == "User Whose Order Got Cancelled":
        system_template = '''
        You are an AI customer support executive for Meesho which is an online e-commerce company, your job is to segment the user query.
        Segments 
        A) User wants to know why their order got cancelled. Eg: Why was my order cancelled?
        B) User wants to know when they'll receive the refund for their cancelled order. Eg: When will I receive the refund for my cancelled order?
        E) Anything else. Context outside Meesho, outside cancelled related queries. Eg: Details of the product, details of the seller, etc. 
        You have to classify the user query into one of the above segments. Only return A,B,C,D or E only. Strictly return any of these 5 characters according to intent in the output and nothing else.
        '''
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(first_input)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    chain = LLMChain(llm=llm, prompt=chat_prompt)
    answer = chain.run({})
    if answer == "E":
        return None, None
    else:
        if scenario_selected == "User Whose Order is Delivered":
            return delivered_sops[answer]
        elif scenario_selected == "User Whose Order Got Cancelled":
            return cancelled_sops[answer]


constitution_prompt = '''You are a customer support executive responsible for resolving user queries for Meesho which is an e-commerce company. Adhere to the following guidelines:
                        1. Always be empathetic and kind to users. If a user uses foul language, remain calm and assist them.
                        2. Use the language the user is using, such as English, Hindi, or Hinglish. Only use the Roman script when responding.
                        3. Avoid using complex or heavy Hindi words. Keep the language a mixture of Hindi and English when the user is using Hindi or Hinglish.
                        4. Keep your responses brief, within two lines, and avoid repeating information.
                        5. If you are unsure about something, admit it instead of making up an answer.
                        6. If a policy is breached and you are unsure how to proceed, repeat the policy, apologize, and state that feedback has been taken.
                        7. Introduce yourself as a Human Support Agent with name Deepak. Do not reveal that you are an AI.
                        8. Respond in a respectful, inclusive, and unbiased language, avoiding discrimination based on factors like caste, creed, gender, or religion. Maintain composure despite any provocation.
                        9. Maintain a professional and courteous tone, avoiding casual language.
                        10. Provide prompt and efficient assistance to respect the customer's time, minimizing unnecessary delays.
                        11. Communicate information clearly and transparently, avoiding jargon or technical terms that may confuse the customer.
                        12. Provide accurate and reliable information, ensuring it is up to date and relevant to the customer's query.
                        13. Prioritize the confidentiality and privacy of customer information, sharing sensitive personal details only when necessary for issue resolution.
                        14. You will be getting the standard operating procedure and data for the user query. You have to follow the SOP and answer the user query.'''

if user_input := st.chat_input("Please state your query in detail?"):
    if not scenario:
        st.info("Please select a scenario from the sidebar")
        st.stop()

    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    if "intent_detected" not in st.session_state:
        sop_for_intent = detect_sop_from_query(user_input, scenario)
        if sop_for_intent:
            st.session_state["intent_detected"] = True
            if "sop" not in st.session_state:
                st.session_state["sop"] = sop_for_intent
            if "data" not in st.session_state:
                st.session_state['data'] = f"Payment Mode: {st.session_state['payment_mode']}\n, Order Status: {st.session_state['order_status']}\n, Return Window: {st.session_state['return_window']}\n, Return Type: {st.session_state['return_type']}\n, Refund Status: {st.session_state['refund_status']}\n, One Return Request Cancelled: {st.session_state['one_return_request_cancelled']}\n, Expected Date of Refund: {st.session_state['expected_date_of_refund']}\n, order_num: '12345678'\n, order_id: '12345678'\n, created: '2023-06-27 11:08:08'\n, payment_charge: '318'\n, courier_name: 'Delhivery'\n, user_address: 'lalu kushwaha kirana stor, vill.barehda atarra banda up, lalu kushwaha kirana stor barehda , atarra, Uttar Pradesh'\n, user_name: 'Rohan'\n "
                st.session_state.messages.append({"role": "system",
                                                  "content": f"{constitution_prompt} \n SOP: {st.session_state['sop']} Data: {st.session_state['data']}"})
                with st.chat_message("Info"):
                    st.markdown(f"Data selected :  {st.session_state['data']}")
                st.session_state.messages.append(
                    {"role": "Info", "content": f"Data selected :  {st.session_state['data']}"})
        else:
            st.info(
                "Sorry, I didn't understand the intent of the query or your query is outside the scope of current system's understanding. Please try again.")

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in openai.ChatCompletion.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                    if m["role"] != "Info"
                ],
                stream=True,
        ):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
