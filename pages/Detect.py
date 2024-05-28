import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time
from joblib import load
import asyncio
import seaborn as sns
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer
from sklearn.compose import ColumnTransformer

# Page title and configuration


def main():
    st.set_page_config(page_title="Fire Detection System", layout="wide")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard",  "Test", "Alarm logs"])

    st.title("Fire Detection System")
    if page == "Dashboard":
        st.subheader("Dashboard Overview")
        st.markdown("----")
        col1,col2 = st.columns(2)
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
        with col1:
            st.write("""
                     ##### Location A""")

        with col2:
            st.success("Alarm")
            
       
        def generate_data():
            data = {
            'Temperature': np.random.uniform(20, 30),
            'eCO2': np.random.uniform(400, 600),
            'Pressure': np.random.uniform(1000, 1020),
            'Raw_H2': np.random.uniform(0.1, 0.2),
            'NC2.5': np.random.uniform(10, 20),
            'CNT': np.random.uniform(100, 200),
            'TVOC': np.random.uniform(200, 300)
            }
            return data
        df = pd.DataFrame(columns=['Temperature', 'eCO2', 'Pressure', 'Raw_H2', 'NC2.5', 'CNT', 'TVOC'])
        data_placeholder = st.empty()
        chart_placeholder = st.empty()
        while True:
    # Generate new data
            new_data = pd.DataFrame([generate_data()])
            df = pd.concat([df, new_data], ignore_index=True)
            if len(df) > 100:
                df = df.iloc[-100:]
            # Display data in a table
            with data_placeholder.container():
                st.write("""
                         ##### Sensor Data""")
                st.dataframe(new_data)
            
            # Create line charts for each sensor
            with chart_placeholder.container():
        
                fig = go.Figure()

                # Temperature
                fig.add_trace(go.Scatter(x=df.index, y=df['Temperature'], mode='lines+markers', name='Temperature'))

                # eCO2
                fig.add_trace(go.Scatter(x=df.index, y=df['eCO2'], mode='lines+markers', name='eCO2'))

                # Pressure
                fig.add_trace(go.Scatter(x=df.index, y=df['Pressure'], mode='lines+markers', name='Pressure'))

                # Raw_H2
                fig.add_trace(go.Scatter(x=df.index, y=df['Raw_H2'], mode='lines+markers', name='Raw_H2'))

                # NC2.5
                fig.add_trace(go.Scatter(x=df.index, y=df['NC2.5'], mode='lines+markers', name='NC2.5'))

                # CNT
                fig.add_trace(go.Scatter(x=df.index, y=df['CNT'], mode='lines+markers', name='CNT'))

                # TVOC
                fig.add_trace(go.Scatter(x=df.index, y=df['TVOC'], mode='lines+markers', name='TVOC'))

                fig.update_layout(
                    title='Sensor Data Over Time',
                    xaxis_title='Time',
                    yaxis_title='Values',
                    template='plotly_white'
                )

                st.plotly_chart(fig, use_container_width=True)
            
            # Wait for a few seconds before updating
            time.sleep(2)
            
            # Clear the output to update the graphs
        st.experimental_rerun()

       
        # try:
        #     asyncio.run(draw_async(time_placeholder,loc1,loc2, loc3, alarm1,alarm2,alarm3))
        # except Exception as e:
        #     print(f'Error: {type(e)}')
        #     raise
        # finally:
        #     print('Finally')      
              
    # Run the time update function in a separate thread
        
        
    elif page == "Test":
    # Layout the app beforehand, with st.empty for the widgets
        st.subheader("Specify Input Parameters")
        col1, col2 = st.columns(2)
        with col1:
            Temp = st.slider('Temperature', 10, 15, 59)
            TVOC = st.slider('TVOC', 0, 1942, 60000)
            Raw_H2 = st.slider('Raw H2', 10000, 13109)
            NC25 = st.slider('NC2.5', 0, 16, 30000)
        with col2:
            CNT = st.slider('CNT', 0, 13, 25000)
            eCO2 = st.slider('eCO2', 400, 60000)
            Pressure = st.slider('Pressure', 930, 940)
            
            data = {'Temperature': Temp,
                'eCO2': eCO2,
                'Pressure': Pressure,
                'Raw_H2': Raw_H2,
                'NC2.5': NC25,
                'CNT': CNT,
                'TVOC': TVOC,}
            df = pd.DataFrame(data, index=[0])
            st.write("""
                     ##### User Input Parameters""")
            st.write(df)
        st.write('---')
        @st.cache_resource
        def load_model():
            model = load('new_model')
            return model
        model = load_model()
        scaler = StandardScaler()
        transformed = scaler.fit_transform(df)
        x_scaled = pd.DataFrame(transformed, columns=df.columns)
        qt = QuantileTransformer(output_distribution='normal')
        const = 1e-8
        X= df + const
        full_pipeline = ColumnTransformer([
            ('Quantile', qt, x_scaled.columns),
        ])
        train_prepared_new = full_pipeline.fit_transform(x_scaled)
        train_prep = pd.DataFrame(train_prepared_new, columns=x_scaled.columns)
       

        if st.button('Submit'):
             y_pred = model.predict(train_prep)
             st.write(y_pred)

    st.sidebar.markdown("---")
    st.sidebar.markdown("© 2024 Fire Detection System")

# async def draw_async(time_placeholder,loc1,loc2, loc3, alarm1,alarm2,alarm3):
#     # while True:
    #         # Get current time
    #     timestamp = datetime.now()
    #     current_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    #     time_placeholder.markdown("Current Time: " + current_time, unsafe_allow_html=True)
        
    #     loc1.markdown("Location A")
    #     loc2.markdown('Location B')
    #     loc3.markdown('Location C')
    #     alarm1.success("Alarm")
    #     alarm2.success("Alarm")
    #     alarm3.success("Alarm")
    #     await asyncio.sleep(1)
if __name__ == '__main__':
    main()