import streamlit as st
import pandas as pd
import altair as alt
from query_engine import load_rag_components, query_rag

# This is a placeholder
@st.cache_data
def load_data():
    """Loads a sample of bank dataset for demonstration."""
    df = pd.read_parquet("./data/domain_aggregates.parquet")
    return df


# Streamlit App Layout

st.set_page_config(
    page_title="Ask My Data",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Ask My Data �")
st.write("Ask questions about the bank dataset using natural language.")

# Load the RAG components just once using Streamlit's caching.
rag_model, index, metadata = load_rag_components()

# If the RAG components fail to load, stop the app and show an error.
if rag_model is None or index is None or metadata is None:
    st.error("RAG model components could not be loaded. Please check your files.")
    st.stop()

# High-Level Dataset Information

df = load_data()
with st.expander("Dataset Overview"):
    st.write("Here is a sample of the data:")
    st.dataframe(df.head(10), use_container_width=True)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message['type'] == 'text':
            st.markdown(message["content"])
        elif message['type'] == 'graph':
            st.altair_chart(message["content"], use_container_width=True)

# Accept user input
if prompt := st.chat_input("What do you want to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt, "type": "text"})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get the model's response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Simple logic to check for a request to plot a graph
            if 'graph' in prompt.lower():
                # Aggregate data for the graph
                quarterly_transactions = df[df['domain'] == 'MEDICAL']
                quarterly_transactions = quarterly_transactions.groupby('date')['value'].sum().reset_index()
                
                # Create the Altair chart
                chart = alt.Chart(quarterly_transactions).mark_line(point=True).encode(
                    x=alt.X('date', title='Date'),
                    y=alt.Y('value', title='Aggregated Transactions'),
                    tooltip=[alt.Tooltip('date', title='Date'), alt.Tooltip('value', title='Value', format='$,.2f')]
                ).properties(
                    title='Medical Transaction Trends by Quarter'
                ).interactive()
                
                st.altair_chart(chart, use_container_width=True)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": chart, "type": 'graph'})
            
            else:
                # Use the imported query_rag function for all other queries
                response_text = query_rag(prompt, rag_model, index, metadata)
                st.markdown(response_text)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response_text, "type": 'text'})