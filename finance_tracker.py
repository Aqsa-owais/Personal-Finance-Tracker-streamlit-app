"""
Personal Finance Tracker - Streamlit App
Run with: streamlit run finance_tracker.py

Requirements:
pip install streamlit pandas plotly
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
from dataclasses import dataclass
from typing import List, Optional
import json

# ============== Data Models ==============
@dataclass
class Transaction:
    id: str
    amount: float
    category: str
    transaction_type: str  # 'income', 'expense', 'investment'
    date: str
    note: Optional[str] = None

# ============== Categories ==============
EXPENSE_CATEGORIES = ["Food", "Transport", "Entertainment", "Bills", "Shopping", "Health", "Education", "Other"]
INCOME_CATEGORIES = ["Salary", "Freelance", "Business", "Investments", "Gifts", "Other"]
INVESTMENT_CATEGORIES = ["Stocks", "Crypto", "Real Estate", "Mutual Funds", "Bonds", "Other"]

# ============== Helper Functions ==============
def generate_id() -> str:
    return f"{datetime.now().timestamp()}"

def get_unique_categories(transactions: List[dict]) -> str:
    """Get unique categories with string manipulation"""
    categories = list(set(t['category'] for t in transactions))
    # String manipulation: uppercase first letter, join with separator
    formatted = [cat.upper()[0] + cat[1:].lower() for cat in categories]
    return " | ".join(sorted(formatted))

def calculate_summary(transactions: List[dict]) -> dict:
    total_income = sum(t['amount'] for t in transactions if t['transaction_type'] == 'income')
    total_expenses = sum(t['amount'] for t in transactions if t['transaction_type'] == 'expense')
    total_investments = sum(t['amount'] for t in transactions if t['transaction_type'] == 'investment')
    net_balance = total_income - total_expenses - total_investments
    
    return {
        'total_income': total_income,
        'total_expenses': total_expenses,
        'total_investments': total_investments,
        'net_balance': net_balance
    }

def get_category_breakdown(transactions: List[dict], trans_type: str) -> pd.DataFrame:
    filtered = [t for t in transactions if t['transaction_type'] == trans_type]
    if not filtered:
        return pd.DataFrame()
    
    df = pd.DataFrame(filtered)
    breakdown = df.groupby('category')['amount'].sum().reset_index()
    breakdown.columns = ['Category', 'Amount']
    return breakdown.sort_values('Amount', ascending=False)

def get_insights(transactions: List[dict]) -> dict:
    if not transactions:
        return {}
    
    expenses = [t for t in transactions if t['transaction_type'] == 'expense']
    
    if not expenses:
        return {'unique_categories': get_unique_categories(transactions)}
    
    # Highest spending category
    category_totals = {}
    for t in expenses:
        category_totals[t['category']] = category_totals.get(t['category'], 0) + t['amount']
    
    highest_category = max(category_totals, key=category_totals.get)
    highest_amount = category_totals[highest_category]
    
    # Most frequent category
    category_counts = {}
    for t in expenses:
        category_counts[t['category']] = category_counts.get(t['category'], 0) + 1
    
    most_frequent = max(category_counts, key=category_counts.get)
    frequency = category_counts[most_frequent]
    
    return {
        'highest_category': highest_category,
        'highest_amount': highest_amount,
        'most_frequent': most_frequent,
        'frequency': frequency,
        'unique_categories': get_unique_categories(transactions)
    }

# ============== Page Config ==============
st.set_page_config(
    page_title="Finance Tracker",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============== Custom CSS for Soft Professional UI ==============
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #faf9f7;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f5f3f0;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #3d3d3d;
        font-weight: 600;
    }
    
    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        border: 1px solid #e8e6e3;
    }
    
    [data-testid="stMetricLabel"] {
        color: #6b6b6b;
        font-size: 14px;
    }
    
    [data-testid="stMetricValue"] {
        color: #3d3d3d;
        font-weight: 600;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #7c9a8e;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background-color: #6a877c;
        box-shadow: 0 4px 12px rgba(124, 154, 142, 0.3);
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div,
    .stDateInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #e0ded9;
        background-color: #ffffff;
    }
    
    /* Cards/Containers */
    .css-1r6slb0, .css-12oz5g7 {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #e8e6e3;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #7c9a8e;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #ffffff;
        border-radius: 8px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 10px 20px;
        border: 1px solid #e8e6e3;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #7c9a8e;
        color: white;
    }
    
    /* Dataframe */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Success/Info/Warning messages */
    .stSuccess, .stInfo, .stWarning {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ============== Session State Initialization ==============
if 'transactions' not in st.session_state:
    st.session_state.transactions = []

if 'savings_goal' not in st.session_state:
    st.session_state.savings_goal = 20  # Default 20%

# ============== Sidebar - Add Transaction ==============
with st.sidebar:
    st.markdown("## Add Transaction")
    
    transaction_type = st.selectbox(
        "Type",
        ["expense", "income", "investment"],
        format_func=lambda x: x.capitalize()
    )
    
    # Dynamic category selection based on type
    if transaction_type == "expense":
        categories = EXPENSE_CATEGORIES
    elif transaction_type == "income":
        categories = INCOME_CATEGORIES
    else:
        categories = INVESTMENT_CATEGORIES
    
    category = st.selectbox("Category", categories)
    amount = st.number_input("Amount (Rs.)", min_value=0.0, step=100.0, format="%.2f")
    trans_date = st.date_input("Date", value=date.today())
    note = st.text_input("Note (optional)")
    
    if st.button("Add Transaction", use_container_width=True):
        if amount > 0:
            new_transaction = {
                'id': generate_id(),
                'amount': amount,
                'category': category,
                'transaction_type': transaction_type,
                'date': trans_date.strftime("%Y-%m-%d"),
                'note': note if note else None
            }
            st.session_state.transactions.append(new_transaction)
            st.success(f"Added {transaction_type}: Rs. {amount:,.2f}")
            st.rerun()
        else:
            st.warning("Please enter an amount greater than 0")
    
    st.markdown("---")
    
    # Savings Goal Setting
    st.markdown("## Savings Goal")
    new_goal = st.slider(
        "Monthly savings target (%)",
        min_value=5,
        max_value=50,
        value=st.session_state.savings_goal,
        step=5
    )
    if new_goal != st.session_state.savings_goal:
        st.session_state.savings_goal = new_goal

# ============== Main Content ==============
st.markdown("# Personal Finance Tracker")
st.markdown("*Track your income, expenses, and investments with ease*")

# Calculate summary
summary = calculate_summary(st.session_state.transactions)

# ============== Summary Cards ==============
st.markdown("### Financial Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Income",
        value=f"Rs. {summary['total_income']:,.0f}",
        delta=None
    )

with col2:
    st.metric(
        label="Total Expenses",
        value=f"Rs. {summary['total_expenses']:,.0f}",
        delta=None
    )

with col3:
    st.metric(
        label="Investments",
        value=f"Rs. {summary['total_investments']:,.0f}",
        delta=None
    )

with col4:
    balance_color = "normal" if summary['net_balance'] >= 0 else "inverse"
    st.metric(
        label="Net Balance",
        value=f"Rs. {summary['net_balance']:,.0f}",
        delta=f"{'Surplus' if summary['net_balance'] >= 0 else 'Deficit'}"
    )

st.markdown("---")

# ============== Savings Goal Progress ==============
if summary['total_income'] > 0:
    st.markdown("### Savings Goal Progress")
    
    savings = summary['total_income'] - summary['total_expenses']
    savings_percentage = (savings / summary['total_income']) * 100 if summary['total_income'] > 0 else 0
    goal_percentage = st.session_state.savings_goal
    progress = min(savings_percentage / goal_percentage, 1.0) if goal_percentage > 0 else 0
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.progress(max(0, progress))
        
    with col2:
        if savings_percentage >= goal_percentage:
            st.success(f"Goal reached! {savings_percentage:.1f}%")
        else:
            st.info(f"{savings_percentage:.1f}% / {goal_percentage}%")
    
    st.caption(f"You've saved Rs. {savings:,.0f} ({savings_percentage:.1f}% of income). Target: {goal_percentage}%")
    
    st.markdown("---")

# ============== Charts Section ==============
if st.session_state.transactions:
    st.markdown("### Spending Analytics")
    
    tab1, tab2, tab3 = st.tabs(["Expenses", "Income", "Investments"])
    
    with tab1:
        expense_breakdown = get_category_breakdown(st.session_state.transactions, 'expense')
        if not expense_breakdown.empty:
            fig = px.bar(
                expense_breakdown,
                x='Category',
                y='Amount',
                color='Amount',
                color_continuous_scale=['#a8c5b8', '#7c9a8e', '#5a7d6e'],
                title=None
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#3d3d3d'),
                showlegend=False,
                coloraxis_showscale=False
            )
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No expense transactions yet")
    
    with tab2:
        income_breakdown = get_category_breakdown(st.session_state.transactions, 'income')
        if not income_breakdown.empty:
            fig = px.pie(
                income_breakdown,
                values='Amount',
                names='Category',
                color_discrete_sequence=['#7c9a8e', '#a8c5b8', '#c9ddd3', '#5a7d6e', '#8fb3a3', '#d4e6dc']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#3d3d3d')
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No income transactions yet")
    
    with tab3:
        investment_breakdown = get_category_breakdown(st.session_state.transactions, 'investment')
        if not investment_breakdown.empty:
            fig = px.pie(
                investment_breakdown,
                values='Amount',
                names='Category',
                hole=0.4,
                color_discrete_sequence=['#c9a87c', '#d4bc9a', '#b89660', '#e0d0b8', '#a68a5c']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#3d3d3d')
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No investment transactions yet")
    
    st.markdown("---")
    
    # ============== Insights Panel ==============
    st.markdown("### Financial Insights")
    insights = get_insights(st.session_state.transactions)
    
    if insights:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'highest_category' in insights:
                st.markdown("**Highest Spending**")
                st.markdown(f"📊 {insights['highest_category']}")
                st.caption(f"Rs. {insights['highest_amount']:,.0f}")
        
        with col2:
            if 'most_frequent' in insights:
                st.markdown("**Most Frequent**")
                st.markdown(f"🔄 {insights['most_frequent']}")
                st.caption(f"{insights['frequency']} transactions")
        
        with col3:
            st.markdown("**Categories Used**")
            st.markdown(f"📁 {len(set(t['category'] for t in st.session_state.transactions))}")
            st.caption(insights.get('unique_categories', 'N/A'))
    
    st.markdown("---")
    
    # ============== Transaction History ==============
    st.markdown("### Transaction History")
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        filter_type = st.selectbox(
            "Filter by type",
            ["All", "Income", "Expense", "Investment"]
        )
    
    # Filter transactions
    filtered_transactions = st.session_state.transactions.copy()
    if filter_type != "All":
        filtered_transactions = [t for t in filtered_transactions if t['transaction_type'] == filter_type.lower()]
    
    # Sort by date (newest first)
    filtered_transactions.sort(key=lambda x: x['date'], reverse=True)
    
    if filtered_transactions:
        for i, trans in enumerate(filtered_transactions):
            with st.expander(
                f"{'🟢' if trans['transaction_type'] == 'income' else '🔴' if trans['transaction_type'] == 'expense' else '🟡'} "
                f"{trans['category']} - Rs. {trans['amount']:,.0f} | {trans['date']}"
            ):
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.write(f"**Type:** {trans['transaction_type'].capitalize()}")
                    st.write(f"**Amount:** Rs. {trans['amount']:,.2f}")
                
                with col2:
                    st.write(f"**Category:** {trans['category']}")
                    st.write(f"**Date:** {trans['date']}")
                
                with col3:
                    if st.button("Delete", key=f"del_{trans['id']}"):
                        st.session_state.transactions = [t for t in st.session_state.transactions if t['id'] != trans['id']]
                        st.rerun()
                
                if trans['note']:
                    st.caption(f"Note: {trans['note']}")
    else:
        st.info("No transactions found")

else:
    st.info("👋 Welcome! Add your first transaction using the sidebar to get started.")

# ============== Footer ==============
st.markdown("----")
st.caption("Personal Finance Tracker | Built with Streamlit")
