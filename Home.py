import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_curve, auc, precision_recall_curve, average_precision_score

st.set_page_config(
    page_title="Home")
alert = st.sidebar.success('Page loaded successfully')

st.write("""
         # Fire Detection App
         
         This app uses Support Vector Classifier model to detect fire outbreak and provide a time series
         analysis of the trend of fire occurences 
         """)

st.markdown(
"""
------
### System Architecture""")

st.image('sblock.jpeg', caption='Overall System Architecture')
st.markdown(
"""
### Dataset Exploration""")
with st.expander(("See Dataset")):
    ### Dataset used for the Project

    data = pd.read_csv('data.csv')
    st.dataframe(data, use_container_width = True)

with st.expander(("About Dataset")):
    st.markdown(
"""
The dataset features in detail:

1. Air Temperature
2. Air Humidity
3. TVOC: Total Volatile Organic Compounds; measured in parts per billion (Source)
4. eCO2: co2 equivalent concentration; calculated from different values like TVCO
5. Raw H2: raw molecular hydrogen; not compensated (Bias, temperature, etc.)
6. Raw Ethanol: raw ethanol gas (Source)
7. Air Pressure
8. PM 1.0 and PM 2.5: particulate matter size < 1.0 µm (PM1.0). 1.0 µm < 2.5 µm (PM2.5)
9. Fire Alarm: ground truth is "1" if a fire is there
10. CNT: Sample counter
11. UTC: Timestamp UTC seconds
12. NC0.5/NC1.0 and NC2.5: Number concentration of particulate matter.
            The raw NC is also classified by the particle size: < 0.5 µm (NC0.5);
             0.5 µm < 1.0 µm (NC1.0); 1.0 µm < 2.5 µm (NC2.5);
""")
st.markdown("""
###### Plots""")
col1, col2 = st.columns(2)

with col1:
    labels = 'Fire', 'No Fire'
    sizes = data['Target'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Target Variable Distribution')
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig1)

with col2:
    x_axis = st.selectbox('Select X-axis feature:', options=data.columns[:-1])
    y_axis = 'Target'
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=data, x=x_axis, y=y_axis, ax=ax)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_title(f'Scatter Plot of {y_axis} vs {x_axis}')
    st.pyplot(fig)

st.markdown("""
            ------
            ### Model Training Result
            For this system, Support Vector Classifier was trained on the data 
            and the following results were obtained
            """) 

with open('model_pkl' , 'rb') as f:
    model = pickle.load(f)

X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')
test_x = X_test.values
y_pred = model.predict(test_x)
y_scores = model.predict_proba(test_x)

fpr, tpr, _ = roc_curve(y_test, y_scores[:,1])
roc_auc = auc(fpr, tpr)
cm = confusion_matrix(y_test,y_pred)

# st.subheader("Precision-Recall Curve")
fig, ((ax4, ax1), (ax3, ax2)) = plt.subplots(2, 2, figsize=(12, 10))
precision, recall, _ = precision_recall_curve(y_test, y_scores[:,1])
average_precision = average_precision_score(y_test, y_scores[:,1])
# Bar chart for evaluation metrics
metrics = ['Accuracy', 'F1-score', 'Precision', 'Recall']
values = [1.0, 0.99, 0.99, 0.99]  # Replace with your actual metric values
ax4.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
ax4.set_ylim(0, 1)
ax4.set_ylabel('Score')
ax4.set_title('Evaluation Metrics')

# Confusion matrix
sns.heatmap(cm, annot=True, fmt='g', ax=ax1, cmap='Blues')
ax1.set_xlabel('Predicted labels')
ax1.set_ylabel('True labels')
ax1.set_title('Confusion Matrix')
ax1.xaxis.set_ticklabels(['Detected', 'Not detected'])
ax1.yaxis.set_ticklabels(['Detected', 'Not detected'])

# ROC curve
ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('Receiver Operating Characteristic')
ax2.legend(loc="lower right")

# Precision-Recall curve
ax3.plot(recall, precision, color='green', lw=2, label=f'AP = {average_precision:.2f}')
ax3.set_xlim([0.0, 1.0])
ax3.set_ylim([0.0, 1.05])
ax3.set_xlabel('Recall')
ax3.set_ylabel('Precision')
ax3.set_title('Precision-Recall Curve')
ax3.legend(loc="lower left")

fig.subplots_adjust(hspace=0.3)
st.pyplot(fig)
# plot_precision_recall_curve(model, x_test, y_test)
# st.pyplot()

time.sleep(3)
alert.empty()