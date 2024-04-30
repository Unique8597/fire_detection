import asyncio
import pandas as pd
import plotly.express as px

import streamlit as st
from datetime import datetime

def main():
    # Layout the app beforehand, with st.empty for the widgets
    time_placeholder = st.empty()
    col1,col2 = st.columns(2)
    with col1:
        loc1 = st.empty()
    # alarm1 = st.empty()
    with col2:
        loc2 = st.empty()
    # alarm2 = st.empty()
    # loc3 = st.empty()
    # alarm3 = st.empty()


    try:
        # Async run the draw function, sending in all the
        # widgets it needs to use/populate
        asyncio.run(draw_async(time_placeholder,loc1,loc2))
    except Exception as e:
        print(f'Error: {type(e)}')
        raise
    finally:
        print('Finally')

async def draw_async(time_placeholder,loc1,loc2):
    while True:
        # Get current time
        timestamp = datetime.now()
        current_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")

        graph.markdown(current_time)
        
        loc1.markdown("Location A")
        loc2.markdown("Alarm")
        # Wait for 1 second before updating again
        await asyncio.sleep(1)
if __name__ == '__main__':
    main()
