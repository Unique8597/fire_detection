import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time
import asyncio

# Page title and configuration


def main():
    st.set_page_config(page_title="Fire Detection System", layout="wide")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Sensor Status", "Alarm Log", "Settings"])

    st.title("Fire Detection System")

    if page == "Dashboard":
        st.subheader("Dashboard Overview")
        time_placeholder = st.empty()
        st.markdown("----")
        col1,col2 = st.columns(2)
        with col1:
            loc1 = st.empty()
            st.markdown(".")
            loc2 = st.empty()
            st.markdown(".")
            loc3 = st.empty()
        with col2:
            alarm1 = st.empty()
            alarm2 = st.empty()
            alarm3 = st.empty()
        st.write(
        """
        <style>
        .element-container {
            display: flex;
            flex-direction: column;
            align-items: center; /* Change to 'flex-start' or 'flex-end' to adjust alignment */
        }
        </style>
        """,
        unsafe_allow_html=True
        )
       
       
        try:
            asyncio.run(draw_async(time_placeholder,loc1,loc2, loc3, alarm1,alarm2,alarm3))
        except Exception as e:
            print(f'Error: {type(e)}')
            raise
        finally:
            print('Finally')      
              
    # Run the time update function in a separate thread
        


    elif page == "Sensor Status":
        st.subheader("Sensor Status")


    elif page == "Alarm Log":
        st.subheader("Alarm Log")
        st.write("Recent alarms or alerts go here.")

    elif page == "Settings":
        st.subheader("Settings and Configuration")
        st.write("System settings and configuration options go here.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("© 2024 Fire Detection System")

async def draw_async(time_placeholder,loc1,loc2, loc3, alarm1,alarm2,alarm3):
    while True:
            # Get current time
        timestamp = datetime.now()
        current_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        time_placeholder.markdown("Current Time: " + current_time, unsafe_allow_html=True)
        
        loc1.markdown("Location A")
        loc2.markdown('Location B')
        loc3.markdown('Location C')
        alarm1.success("Alarm")
        alarm2.success("Alarm")
        alarm3.success("Alarm")
        await asyncio.sleep(1)
if __name__ == '__main__':
    main()