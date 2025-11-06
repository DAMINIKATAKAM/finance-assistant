import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date, timedelta
from sklearn.linear_model import LinearRegression
import google.generativeai as genai
import os

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Smart Finance Assistant", layout="wide")
DATA_FILE = "transactions_saved.csv"

# ------------------ LOAD TRANSACTIONS ------------------
if os.path.exists(DATA_FILE):
    try:
        df = pd.read_csv(DATA_FILE)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
            df = df.dropna(subset=['Date'])
    except Exception as e:
        st.error(f"Error loading saved data: {e}")
        df = pd.DataFrame(columns=["Date", "Type", "Description", "Category", "Amount"])
else:
    df = pd.DataFrame(columns=["Date", "Type", "Description", "Category", "Amount"])
    df.to_csv(DATA_FILE, index=False)

st.session_state.transactions = df.copy()

# ------------------ SIDEBAR ------------------
st.sidebar.header("User Inputs")

income = st.sidebar.number_input("Monthly Income (₹)", min_value=0.0, step=100.0)
savings_goal = st.sidebar.number_input("Savings Goal (₹)", min_value=0.0, step=500.0)
goal_deadline = st.sidebar.date_input("Goal Deadline", value=date.today())

st.sidebar.subheader("Add a Transaction")
t_date = st.sidebar.date_input("Date", value=date.today())
t_type = st.sidebar.selectbox("Type", ["Debit (Expense)", "Credit (Income)"])
t_desc = st.sidebar.text_input("Description")
t_cat = st.sidebar.selectbox("Category", ["Food", "Transport", "Shopping", "Bills", "Salary", "Other"])
t_amt = st.sidebar.number_input("Amount (₹)", min_value=0.0, step=100.0)

# Save transaction
if st.sidebar.button("Add Transaction"):
    new_row = pd.DataFrame({
        "Date": [pd.to_datetime(t_date)],
        "Type": [t_type],
        "Description": [t_desc],
        "Category": [t_cat],
        "Amount": [t_amt]
    })
    st.session_state.transactions = pd.concat([st.session_state.transactions, new_row], ignore_index=True)
    st.session_state.transactions.to_csv(DATA_FILE, index=False)
    st.sidebar.success("✅ Transaction Added and Saved!")

# ------------------ MAIN PANEL ------------------
st.title("💰 Smart Personal Finance Assistant")

# Show transactions
st.subheader("📋 Your Transactions")
st.dataframe(st.session_state.transactions)

# ------------------ SPENDING TREND ------------------
if not st.session_state.transactions.empty:
    df = st.session_state.transactions.copy()
    df["Date"] = pd.to_datetime(df["Date"], format='mixed', errors='coerce')
    df = df.dropna(subset=['Date'])
    df.sort_values("Date", inplace=True)

    expense_df = df[df["Type"].str.lower().str.contains("debit")]
    if not expense_df.empty:
        daily_expense = expense_df.groupby("Date")["Amount"].sum().reset_index()
        recent_expense = daily_expense.tail(25)

        st.subheader("📊 Daily Spending Trend (Last 25 Days)")
        fig = px.line(recent_expense, x="Date", y="Amount", title="Spending Over Time", markers=True)
        st.plotly_chart(fig, use_container_width=True)

        # ------------------ LINEAR REGRESSION PREDICTION ------------------
        st.subheader("📈 Next Month Expense Prediction (Based on Last 3 Months)")

        # Convert to monthly total expenses
        expense_df["Month"] = expense_df["Date"].dt.to_period("M")
        monthly_expense = expense_df.groupby("Month")["Amount"].sum().reset_index()
        monthly_expense["Month"] = monthly_expense["Month"].astype(str)

        if len(monthly_expense) >= 3:
            X = np.arange(len(monthly_expense)).reshape(-1, 1)
            y = monthly_expense["Amount"].values

            model = LinearRegression()
            model.fit(X, y)

            # Predict next month
            next_index = np.array([[len(monthly_expense)]])
            predicted_expense = model.predict(next_index)[0]

            next_month = pd.Period(monthly_expense["Month"].iloc[-1], freq="M") + 1

            # Combine data for visualization
            future_df = pd.DataFrame({
                "Month": [next_month.strftime("%Y-%m")],
                "Predicted Expense": [predicted_expense]
            })
            combined_df = pd.concat([monthly_expense, future_df.rename(columns={"Predicted Expense": "Amount"})])

            # Plot graph
            fig2 = px.bar(
                combined_df,
                x="Month",
                y="Amount",
                title="Monthly Expense and Next Month Prediction",
                color=combined_df["Month"].isin([next_month.strftime("%Y-%m")]),
                color_discrete_map={True: "orange", False: "blue"}
            )
            st.plotly_chart(fig2, use_container_width=True)

            st.success(f"📌 Predicted Total Expense for {next_month.strftime('%B %Y')}: ₹{predicted_expense:,.2f}")

        else:
            st.info("Need at least 3 months of expense data to predict next month’s spending.")
    else:
        st.info("No expense transactions yet. Add a Debit transaction from the sidebar.")

# ------------------ BUDGET SUMMARY ------------------
st.subheader("📌 Budget Summary")
if income > 0 and not st.session_state.transactions.empty:
    df = st.session_state.transactions
    expenses = df[df['Type'].str.contains("Debit")]['Amount'].sum()
    income_vals = df[df['Type'].str.contains("Credit")]['Amount'].sum()
    avg_expense = expenses / 30 if expenses > 0 else 0

    remaining_balance = income + income_vals - expenses
    st.metric("Total Expenses", f"₹{expenses:,.2f}")
    st.metric("Average Daily Expense", f"₹{avg_expense:,.2f}")
    st.metric("Remaining Balance", f"₹{remaining_balance:,.2f}")
else:
    st.info("Enter income and at least one transaction to see budget metrics.")

# ------------------ SAVINGS GOAL ------------------
st.subheader("🎯 Savings Goal Progress")

if savings_goal > 0 and income > 0:
    total_income = df[df["Type"].str.contains("Credit")]["Amount"].sum()
    total_expenses = df[df["Type"].str.contains("Debit")]["Amount"].sum()

    available_balance = income + total_income - total_expenses
    saved_amount = max(available_balance, 0)
    progress = min((saved_amount / savings_goal) * 100, 100)

    st.progress(progress / 100)
    st.metric("Progress", f"{progress:.2f}%")
    st.metric("Amount Saved", f"₹{saved_amount:,.2f}")
    st.metric("Amount Remaining", f"₹{max(savings_goal - saved_amount, 0):,.2f}")

    days_left = (goal_deadline - date.today()).days
    if days_left > 0:
        daily_saving_needed = max((savings_goal - saved_amount) / days_left, 0)
        st.info(f"💡 Save ₹{daily_saving_needed:,.2f} per day to reach your goal.")
    else:
        st.warning("⚠️ The goal deadline has passed.")
else:
    st.info("Enter income and savings goal to track progress.")

# ------------------ ANOMALY DETECTION ------------------
st.subheader("⚠️ Anomaly Highlights")
if not st.session_state.transactions.empty:
    expense_df = df[df["Type"].str.lower().str.contains("debit")]
    if len(expense_df) > 5:
        mean_val = expense_df["Amount"].mean()
        std_val = expense_df["Amount"].std()
        threshold = mean_val + 2 * std_val
        anomalies = expense_df[expense_df["Amount"] > threshold]

        if not anomalies.empty:
            st.warning("🚨 High-spending anomalies detected:")
            st.dataframe(anomalies)
        else:
            st.success("✅ No anomalies detected.")
    else:
        st.info("Add more transactions for anomaly detection.")
else:
    st.info("No data available for anomaly detection.")

# ------------------ GEMINI AI ADVICE ------------------
st.subheader("🤖 Gemini Financial Advice")

api_key = "AIzaSyCBlInGbeaQTkKTPczDH4IF8qIXbC13o3M"  # 🔐 Replace manually
if api_key.strip() != "":
    genai.configure(api_key=api_key)

    if st.button("💬 Get Advice from Gemini"):
        try:
            total_expense = df[df['Type'].str.contains("Debit")]['Amount'].sum()
            total_income = income + df[df['Type'].str.contains("Credit")]['Amount'].sum()
            balance = total_income - total_expense

            prompt = (
                f"My total income is ₹{total_income:.2f}, my total expenses are ₹{total_expense:.2f}, "
                f"and my savings goal is ₹{savings_goal:.2f}. Give me short 4-5 lines of practical budgeting advice."
            )

            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt)
            if response and hasattr(response, 'text'):
                st.info(f"🤖 Gemini says:\n\n{response.text.strip()}")
            else:
                st.warning("⚠️ No response received from Gemini.")
        except Exception as e:
            st.error(f"⚠️ Gemini API Error: {e}")
else:
    st.info("Enter your Gemini API key in the code to enable advice.")

