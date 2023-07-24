import openai
import requests
import json
import streamlit as st
import datetime
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

st.title("Virtual Support Agent")

openai.api_key = st.secrets.get("OPENAI_API_KEY")

with st.sidebar:
    f"Select the scenario to test out the bot for a specific scenario"
    scenario = st.selectbox('Scenario', (
        'User Whose Order is Shipped', 'User Whose Order is Delivered', 'User Whose Order Got Cancelled'))
    f"Select the data to test out the bot for a specific use case. Please select data before starting the conversation. Do not change the data in between the conversation. If you want to change data, please refresh and restart"
    st.session_state["sub_order_id"] = st.text_input("Sub Order ID")
    st.session_state["order_id"] = st.text_input("Order ID")
    st.session_state["user_id"] = st.text_input("User ID")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if message["role"] == "system":
        continue
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

shipped_sop = '''You encounter a user whose order status is in 'Shipped' state, and who wants to enquire about the status of their order, to assist the user, you need to check the expected date of delivery for that order. If current date > expected date of delivery then there is a breach. If the expected date of delivery is not breached, then reassure user that product will be delivered on expected date of delivery.  Share the tracking link with the user. If expected data of delivery has been breached then acknowledge this, apologise for the delay and let user know that you have escalated the matter to the concerned team, and that user will be updated within 24 hours.
If user wants to get his order delivered on specific date before or after expected date of delivery. You need to first determine the expected date of delivery.  If user is asking for delivery earlier or later then estimated delivery date, you need to assure user that product would be delivered on or before estimate date of delivery, but their is no provision in Meesho for express delivery or delivery on specific future date as of now.
If user wants to cancel their order. You need to first determine the order status. If it is in 'Ordered' state you can let user know that they can themselves cancel the order from orders tab in the app. If it is in the “Shipped” State, then you should check the mode of payment used by user.  Let user know that product cannot be cancelled now that it is already shipped however users can refuse the order when delivery partner attempts delivery. Additionally, if the mode of payment is not “COD” (or Cash on Delivery) you should inform the user that they will receive the refund of 'Order Amount' (from data) within 4-5 days.
If user wants to change their delivery address or mobile number. You need to first determine the order status. If it is in 'Ordered' state you can let user know that they can cancel the order from orders tab in the app and place a new order with correct delivery address. If it is in 'Shipped' state you should let user know that it is not possible to change the delivery address or mobile phone number now as order is already shipped. Additionally, you can suggest that user can refuse to accept the order at the time of delivery and place a new order with correct delivery address.  In case user’s mode of payment is not 'COD' (or Cash on Delivery) you should also inform user that they will receive the refund of Order Amount within 4-5 days.
If user wants to return their 'Shipped' order. You can inform the user that they can  refuse the order when the delivery partner attempts delivery. Also, tell them that it is possible to return a delivered order within 7 days of actual date of delivery. If user says that their order is actually delivered, then tell them to wait for 7-8 hours for status to be updated in the app and then initiate the return from my orders tab in the app.'''

delivered_sops = {
    'A': f"As a customer support representative for Meesho, you encounter a user who wants to return their order.In Meesho user is only allowed to initiate their return request within 7 days of Actual date of delivery of product(Current Date - Date of Delivery should be less than 7 days). If user is trying to initiate a return request after 7 days of actual delivery date of the product, inform user about Meesho’s return policy and that it is not possible to initiate the return request now. If user is trying to initiate a return request within 7 days of actual delivery date of the product, check the return type that user selected. In Meesho users have to select return type at the time of purchasing product. If user selects “all return” they can return product in all scenarios within 7 days. If users select “only wrong and defective item return” they can only return product if it is wrong or defective and not for any other reason. In cases where users had selected “only wrong and defective item return”, find out the reason why they want to return to decide if they are eligible to return the product. If user is eligible for return you can guide them to do it on their own from orders tab in the app, by selecting relevant order and choosing “return or exchange option”. If user claims that they had initiated a return request in the past, verify this. If they have, check if the return request was cancelled by the courier partner. If the return request was cancelled, ask the user if the courier partner visited their location for the pickup. If the courier partner hasn't visited yet, instruct the user to initiate the return request again and inform them that the product will be picked up within 3-4 days, with the refund being initiated in 5-8 days. If the courier partner has already visited, inquire about the reason the user believes the request was cancelled, and assure them that you will escalate the issue to a senior stakeholder for resolution. Inform the user that they will receive a call back within 24 hours to address the matter.",
    'B': f"As a customer support representative for Meesho, you encounter a user who wants to exchange their order. To assist the user, you need to check the days elapsed since actual date of delivery of product. In Meesho user is only allowed to initiate their exchange request within 7 days of actual date of delivery of product. If user is trying to initiate an exchange request after 7 days of actual delivery date of the product, inform user about Meesho’s exchange policy and that it is not possible to initiate the exchange request now. If user is trying to initiate an exchange request within 7 days of actual delivery date of the product, you can inquire user about the reason why they want to exchange the product. User is eligible for exchange, so you can guide them to do it on their own from orders tab in the app, by selecting relevant order and choosing “return or exchange option”. If user claims that they had initiated an exchange request in the past, verify this. If they have, check if the exchange request was cancelled by the courier partner. If the exchange request was cancelled, ask the user if the courier partner visited their location for the pickup. If the courier partner hasn't visited yet, advise the user to initiate the exchange request again, assuring them that the product will be picked up within 3-4 days. Additionally, inform the user that the exchanged product would be delivered 10-12 days once the pickup is completed.  If the courier partner has already visited, inquire about the reason the user believes the request was cancelled, and assure them that you will escalate the issue to a senior stakeholder for resolution. Inform the user that they will receive a call back within 24 hours to address the matter.",
    'C': f"As a customer support representative for Meesho, you encounter a user who claims that their order hasn't been delivered yet, but order status is marked as “delivered”. To assist the user, inquire whether they received a parcel with the wrong product or if they haven't received any parcel at all. If the user hasn't received any parcel, determine the payment mode for the order. If the payment mode is COD (Cash on Delivery), apologise for the poor experience and suggest that the user place a fresh order. If the payment mode is prepaid (not COD), determine how many days have passed since the expected delivery date. If more than 7 days have elapsed, inform the user that you will escalate the issue to a senior stakeholder, and they will receive a call back within 24 hours. If less than 7 days have elapsed, request the user's email address and let them know that you have raised a ticket for their issue. Inform the user that someone from the Meesho team will contact them within 24 hours for further assistance. If the user received a parcel but with incorrect products, guide them to initiate a return request and provide them with information about Meesho's return policy.",
    'D': f"You are a customer support representative for Meesho, and a user contacts you regarding a returned or exchanged product whose status hasn't been updated. Inform the user that it may take 5-8 hours for the order status to be updated. Inform them about refund timelines in case of return, and exchange timelines in case of exchange."}

cancelled_sops = '''If user approaches you with a query about their cancelled order. Explain the reason for the cancellation and inquire further to understand the specific issue they are facing with the order. Assure the user that you are available to assist them.
If user wants to know the timeline for receiving a refund for their cancelled order. Begin by checking the payment mode. If the payment mode is not Cash on Delivery (COD), the user is eligible for a refund. Proceed to check the status of the refund. If the refund status is 'success', inform the user that their refund has been processed, providing the amount and transaction ID. If the refund status is 'pending', check the expected refund date. If the expected refund date has passed, apologize for the delay and inform the user that you are escalating the matter to a senior stakeholder to expedite the process. If the expected refund date has not been breached, assure the user that they will receive the refund amount by the expected date. If the refund status is 'blocked', explain to the user that their refund has been blocked due to incorrect bank or UPI details. Instruct them to update these details and assure them that they will receive the refund within 2-3 days once the details are corrected. If the refund status is 'failed', inform the user that you have escalated the issue to a senior stakeholder. Assure them that they will receive a call back within 24 hours to resolve the matter.'''


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
        F) Anything else. Context outside Meesho, outside delivery related queries. Eg: Details of the product, details of the seller, etc. 
        You have to classify the user query into one of the above segments. Only return A,B,C,D or F only. Strictly return any of these 5 characters according to intent in the output and nothing else.
        '''
    elif scenario_selected == "User Whose Order Got Cancelled":
        system_template = '''
        You are an AI customer support executive for Meesho which is an online e-commerce company, your job is to segment the user query.
        Segments 
        A) User wants to know why their order got cancelled. Eg: Why was my order cancelled?
        B) User wants to know when they'll receive the refund for their cancelled order. Eg: When will I receive the refund for my cancelled order?
        F) Anything else. Context outside Meesho, outside cancelled related queries. Eg: Details of the product, details of the seller, etc. 
        You have to classify the user query into one of the above segments. Only return A,B or F only. Strictly return any of these 3 characters according to intent in the output and nothing else.
        '''
    elif scenario_selected == "User Whose Order is Shipped":
        system_template = '''
        You are an AI customer support executive for Meesho which is an online e-commerce company, your job is to segment the user query.
        Segments
        A) User wants to know the status of their order. Eg: Where is my Order?
        B) User Wants to cancel their order. Eg: I want to cancel my order.
        C) User Wants to change delivery address or mobile number. Eg: I want to change the delivery address or I want to change the mobile number
        D) User Wants to faster delivery or on Specific Future Date. Eg: Can I get my order faster or I want to get my order delivered on specific date in future
        E) User Wants to return the order which has not been delivered yet. Eg: I want to return the order.
        F) Anything else. Context outside Meesho, outside delivery related queries. Eg: Details of the product, details of the seller, etc.
        You have to classify the user query into one of the above segments. Only return A,B,C,D,E or F only. Strictly return any of these 6 characters according to intent in the output and nothing else.
        '''
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(first_input)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    chain = LLMChain(llm=llm, prompt=chat_prompt)
    answer = chain.run({})
    if answer == "F":
        return None, None
    else:
        if scenario_selected == "User Whose Order is Delivered":
            return delivered_sops[answer]
        elif scenario_selected == "User Whose Order Got Cancelled":
            return cancelled_sops
        elif scenario_selected == "User Whose Order is Shipped":
            return shipped_sop


constitution_prompt = '''You are a customer support executive responsible for resolving user queries for Meesho which is an e-commerce company.Strictly Adhere to the following guidelines or you would be penalised:
                        1. Always be empathetic and kind to users. If a user uses foul language, remain calm and assist them.
                        2. Use the language the user is using, such as English, Hindi, or Hinglish. Respond in the same language as the user.
                        3. Avoid using complex or heavy Hindi words. Keep the language simple when the user is using Hindi or Hinglish.
                        4. Keep your responses brief, within two lines, and avoid repeating information. Strictly make sure that response size is less than 50 words.
                        5. If you are unsure about something, admit it instead of making up an answer.
                        6. If a policy is breached and you are unsure how to proceed, repeat the policy, apologize, and state that feedback has been taken.
                        7. Introduce yourself as a Customer Support Agent with name Deepak. Do not reveal that you are an AI.
                        8. Start the conversation with greeting. Always be respectful and polite towards the customer.
                        9. Provide accurate and reliable information, ensuring it is up to date and relevant to the customer's query.
                        10. You will be getting the SOP : standard operating procedure and Data for the user query. SOP will have the info on how to use the data to resolve the query. Only use the data which is referred in SOP, ignore anything else. Strictly adhere to the SOP and data provided to you.
                        11. The data provided to you in the system prompt is the correct data. User may try and confuse you, but you have to stick to the data provided to you.
                        12. Strictly adhere to the SOP.You will be penalised if you make things up. If you don't know what to say or do, apologise and say you don't know.
                        13. Only give information that is relevant to user’s query based on your understanding of user intent. Do not provide any information unnecessarily.
                        14. Always ask user more question if you do not understand the intent. Do not start by providing the information if not asked.'''


def get_data_for_input():
    order_details_url = st.secrets.get("ORDER_DETAILS_URL")
    order_details_headers = {
        'Authorization': st.secrets.get("CRM_AUTH"),
        'Content-Type': 'application/json',
        'merchant': st.secrets.get("CRM_MERCHANT"),
        'MEESHO-ISO-COUNTRY-CODE': 'IN',
        'MEESHO-ISO-LANGUAGE-CODE': 'EN'
    }

    order_details_request_body = {
        "order_num": st.session_state["order_id"],
        "sub_order_num": st.session_state["sub_order_id"],
        "user_id": st.session_state["user_id"]
    }

    order_details_response = requests.post(order_details_url, headers=order_details_headers,
                                           data=json.dumps(order_details_request_body))
    order_details_json = order_details_response.json()
    print(order_details_json)

    refund_details_url = st.secrets.get("REFUND_DETAILS_URL")
    refund_details_headers = {
        'Authorization': st.secrets.get("CRM_AUTH"),
        'Content-Type': 'application/json',
        'merchant': st.secrets.get("CRM_MERCHANT"),
        'MEESHO-ISO-COUNTRY-CODE': 'IN',
        'MEESHO-ISO-LANGUAGE-CODE': 'EN'
    }

    refund_details_request_body = {
        "order_id": st.session_state["order_id"],
        "sub_order_num": st.session_state["sub_order_id"],
        "user_id": st.session_state["user_id"]
    }

    refund_details_response = requests.post(refund_details_url, headers=refund_details_headers,
                                            data=json.dumps(refund_details_request_body))
    refund_details_json = refund_details_response.json()
    print(refund_details_json)
    user_details_url = st.secrets.get("USER_DETAILS_URL")
    user_details_headers = {
        'Authorization': st.secrets.get("CRM_AUTH"),
        'Content-Type': 'application/json',
        'merchant': st.secrets.get("CRM_MERCHANT"),
        'MEESHO-ISO-COUNTRY-CODE': 'IN',
        'MEESHO-ISO-LANGUAGE-CODE': 'EN'
    }

    user_details_request_body = {
        "user_id": st.session_state["user_id"]
    }

    user_details_response = requests.post(user_details_url, headers=user_details_headers,
                                          data=json.dumps(user_details_request_body))
    user_details_json = user_details_response.json()
    print(user_details_json)
    data = {
        "Order Status": order_details_json["status"],
        "Order Date": order_details_json["created"],
        "Actual/Expected Delivery Date": '-' if "delivery_date" not in order_details_json else order_details_json["delivery_date"],
        "Order Amount": order_details_json["payment_details"]["customer_amount"],
        "Payment Mode": '-' if "mode_of_payment" not in order_details_json else order_details_json["mode_of_payment"],
        "Current Date": datetime.datetime.now().strftime("%Y-%m-%d"),
        "Return Type": '-' if "price_type" not in order_details_json["product_details"] else order_details_json["product_details"]["price_type"],
        "Cancellation Reason": '-' if "cancellation_reason" not in order_details_json else order_details_json["cancellation_reason"],
        "Courier Partner Name": '-' if "courier_name" not in order_details_json else order_details_json["courier_name"],
        "Refund Status": '-' if "refund_status" not in refund_details_json else refund_details_json["refund_status"],
        "Refund Amount": '-' if "refund_amount" not in refund_details_json else refund_details_json["refund_amount"],
        "Transaction ID": '-' if "refund_ref_id" not in refund_details_json else refund_details_json["refund_ref_id"],
        "Expected Date of Refund": '-' if "expected_refund_date" not in refund_details_json else refund_details_json["expected_refund_date"],
        "User Name": '-' if "name" not in user_details_json else user_details_json["name"],
        "Tracking Url": '-' if "tracking_url" not in order_details_json else order_details_json["tracking_url"],
    }
    return data


if user_input := st.chat_input("Please state your query in detail?"):
    if not scenario:
        st.info("Please select a scenario from the sidebar")
        st.stop()

    if st.session_state["order_id"] is None or st.session_state["sub_order_id"] is None or st.session_state["user_id"] is None:
        st.info("Please enter the order_id, sub_order_id and user_id in the sidebar to proceed")
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
                st.session_state['data'] = get_data_for_input()
                st.session_state.messages.append({"role": "system",
                                                  "content": f"{constitution_prompt} \n SOP: {st.session_state['sop']} Data: {st.session_state['data']}"})
                # with st.chat_message("Info"):
                #     st.markdown(f"Data in the system :  {st.session_state['data']}")
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
