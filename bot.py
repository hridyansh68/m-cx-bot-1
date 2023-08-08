import requests
import streamlit as st

st.title("Virtual Support Agent")

CHATBOT_ENDPOINT = st.secrets.get("CHATBOT_ENDPOINT")
CHATBOT_AUTHORISATION = st.secrets.get("CHATBOT_AUTHORISATION")

with st.sidebar:
    f"Select the data to test out the bot for a specific use case. Please select data before starting the conversation. Do not change the data in between the conversation. If you want to change data, please refresh and restart"
    st.session_state["sub_order_id"] = st.text_input("Sub Order ID")
    st.session_state["order_id"] = st.text_input("Order ID")
    st.session_state["user_id"] = st.text_input("User ID")
    if "state" in st.session_state:
        f"Current state: {st.session_state['state']}"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "state" not in st.session_state:
    st.session_state["state"] = {}

if "room_code" not in st.session_state:
    st.session_state["room_code"] = "1234"

for message in st.session_state.messages:
    if message["role"] == "system":
        continue
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def fetch_next_message():
    url = CHATBOT_ENDPOINT
    headers = get_headers()

    request = {
        "chat": st.session_state.messages,
        "user_details": {
            "user_id": st.session_state["user_id"],
        },
        "order_details": {
            "order_id": st.session_state["order_id"],
            "sub_order_id": st.session_state["sub_order_id"]
        },
        "state": st.session_state["state"],
        "room_code": st.session_state["room_code"]
    }

    chat_bot_response = requests.post(url=url, headers=headers, json=request)
    if chat_bot_response.status_code == 200 and chat_bot_response.json() is not None:
        return chat_bot_response.json()
    else:
        st.error(f"Error while calling Chatbot API: {request} response: {chat_bot_response.json()}")


def get_headers():
    headers = {
        "Authorization": CHATBOT_AUTHORISATION
    }
    return headers


if user_input := st.chat_input("Please state your query in detail?"):
    if st.session_state["order_id"] is None or st.session_state["sub_order_id"] is None or st.session_state["user_id"] is None:
        st.info("Please enter the order_id, sub_order_id and user_id in the sidebar to proceed")
        st.stop()

    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    assistant_message = fetch_next_message()

    if "finish_chat" in assistant_message and assistant_message["finish_chat"]:
        st.info("Chat completed successfully")
        st.stop()

    if "transfer_to_human_agent" in assistant_message and assistant_message["transfer_to_human_agent"]:
        st.info("Transferring to human agent")
        st.stop()

    if "state" in assistant_message:
        st.session_state["state"] = assistant_message["state"]

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = assistant_message["next_message"]
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
