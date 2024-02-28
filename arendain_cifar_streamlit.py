# Import relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10

# Define the Streamlit app
def app():
    
    st.title('CIFAR-10 Image Classification')
    st.subheader('Modified by Ed Francis Kyle G. Arendain., BSCS 3A')
    
    st.write('Dataset description: CIFAR-10')

    # display choice of classifier
    clf = BernoulliNB() 
    options = ['Naive Bayes', 'Logistic Regression']
    selected_option = st.selectbox('Select the classifier', options)
    if selected_option == 'Logistic Regression':
        clf = LogisticRegression(C=100, max_iter=100, multi_class='auto',
            penalty='l2', random_state=42, solver='lbfgs',
            verbose=0, warm_start=False)
    else:
        clf = BernoulliNB()

    if st.button('Start'):
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()

        # Flatten the images to 1D array
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train.flatten(),
                                                            test_size=0.2, random_state=42)

        clf.fit(X_train, y_train)
        y_test_pred = clf.predict(X_test)

        st.header('Confusion Matrix')
        cm = confusion_matrix(y_test, y_test_pred)
        st.text(cm)
        st.header('Classification Report')
        st.text(classification_report(y_test, y_test_pred))

# run the app
if __name__ == "__main__":
    app()
