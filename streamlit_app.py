import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#Step 1
#Loading the dataset:

df= pd.read_csv("placement.csv")

#Step 2
#Data preproccessing
#Data already preprocessed

#Step 3
#Building our model

x_train,x_test,y_train,y_test=train_test_split(df[['cgpa','placement_exam_marks']],df.placed,test_size=0.2)
model=LogisticRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

def fun():
	st.header("Student Placement Prediction")
	st.info("Enter all the details properly")
	cgpa = st.number_input("Enter your CGPA: ")
	placement_exam_marks= st.number_input("Enter the marks scored in placement exam:")
	li = [cgpa,placement_exam_marks]
	submit=st.button("SUBMIT")
	if submit: 
		output = model.predict([li])
		if output == 1:
			st.success("Yes, You got selected!")
		else:
			st.error("No, you did not get selected!")
fun()
