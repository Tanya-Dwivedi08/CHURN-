import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.express as px

def load_css():
    with open("style.css", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()
# ---------------- CONFIG ----------------
st.set_page_config(page_title="RetainX AI", layout="wide")

# ---------------- CLEAN PREMIUM UI ----------------

# ---------------- LOGIN ----------------
import time   # ✅ 
import time   # ✅ 

import time   # ✅ 

if "login" not in st.session_state:
    st.session_state.login = False

if not st.session_state.login:

    # 🔥 LOGIN UI CARD

    st.markdown("""
<h1 style='
    font-family:'Poppins';
    font-size:45px;
    font-weight:900;
    background: linear-gradient(90deg, #ffffff, #f1f5f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
'>
RetainX
</h1>
<p style='color:#cbd5f5;'>Login to continue</p>
""", unsafe_allow_html=True)

    # 🖼️ LOGO 
    st.image("https://cdn-icons-png.flaticon.com/512/1055/1055687.png", width=80)

    # 🔑 INPUTS
    u = st.text_input("Username", placeholder="Enter username")
    p = st.text_input("Password", type="password", placeholder="Enter password")

    # ✅ REMEMBER ME 
    remember = st.checkbox("Remember me")

    # 🔘 CENTER BUTTON
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        login_btn = st.button("Login")

    # ⚡ LOGIN LOGIC + LOADING
    if login_btn:
        with st.spinner("Logging in..."):   # 
            time.sleep(1)  # fake delay for effect

        if u.strip() == "admin" and p.strip() == "123":
            st.session_state.login = True
            st.rerun()
        else:
            st.error("Invalid credentials")

    st.stop()
# ---------------- DATA ----------------
@st.cache_data
def load():
    df = pd.read_csv("customer_data.csv")
    df['Contract'] = df['Contract'].map({'Month-to-month':0,'One year':1,'Two year':2})
    df['Churn'] = df['Churn'].map({'No':0,'Yes':1})
    return df

data = load()
X = data.drop("Churn", axis=1)
y = data["Churn"]

# ---------------- MODEL ----------------
@st.cache_resource
def model():
    m = RandomForestClassifier(n_estimators=60)
    m.fit(X, y)
    return m

clf = model()
from sklearn.metrics import accuracy_score

y_pred = clf.predict(X)
acc = accuracy_score(y, y_pred)


# ---------------- SIDEBAR ----------------

st.sidebar.markdown("""
<div class="sidebar-title">RetainX</div>
""", unsafe_allow_html=True)

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1055/1055687.png", width=80)

page = st.sidebar.radio("Menu", [
    "📊 Dashboard",
    "🤖 Prediction",
    "👤 Customer Profile",
    "📊 Analytics",
    "📄 Customer Report",
    "🔐 Logout"
])
# ---------------- DASHBOARD ----------------
if page == "📊 Dashboard":

    st.title("📊 Business Dashboard")

    # ---------------- KPI CARDS ----------------
    st.markdown("### 📌 Key Metrics")

    c1, c2, c3, c4 = st.columns(4)

    total_customers = len(data)
    churn_rate = data["Churn"].mean() * 100
    revenue_risk = data[data["Churn"] == 1]["MonthlyCharges"].sum()
    avg_revenue = data["MonthlyCharges"].mean()

    c1.metric("👥 Customers", total_customers)
    c2.metric("📉 Churn Rate", f"{churn_rate:.2f}%")
    c3.metric("💸 Revenue Risk", f"₹{revenue_risk:.0f}")
    c4.metric("💰 Avg Revenue", f"₹{avg_revenue:.0f}")

    # ---------------- FILTERS ----------------
    st.markdown("### 🔍 Filters")

    col1, col2 = st.columns(2)

    with col1:
        contract_filter = st.selectbox("Contract Type", ["All", "Month-to-month", "One year", "Two year"])

    with col2:
        tenure_filter = st.slider("Tenure Range", 1, 72, (1, 72))

    filtered_data = data.copy()

    if contract_filter != "All":
        mp = {"Month-to-month":0,"One year":1,"Two year":2}
        filtered_data = filtered_data[filtered_data["Contract"] == mp[contract_filter]]

    filtered_data = filtered_data[
        (filtered_data["tenure"] >= tenure_filter[0]) &
        (filtered_data["tenure"] <= tenure_filter[1])
    ]

    # ---------------- MAIN CHART ----------------
    st.markdown("### 📈 Spending Trend")

    trend = filtered_data.groupby("tenure")["MonthlyCharges"].mean().reset_index()
    st.plotly_chart(px.line(trend, x="tenure", y="MonthlyCharges"), use_container_width=True)

    # ---------------- SECOND ROW ----------------
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Churn Distribution")
        st.plotly_chart(px.pie(filtered_data, names="Churn", hole=0.5), use_container_width=True)

    with col2:
        st.markdown("### 📉 Churn by Contract")
        contract = filtered_data.groupby("Contract")["Churn"].mean().reset_index()
        st.plotly_chart(px.bar(contract, x="Contract", y="Churn", color="Contract"), use_container_width=True)

    # ---------------- THIRD ROW ----------------
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 💰 Revenue Loss")
        rev = filtered_data[filtered_data["Churn"] == 1]
        st.plotly_chart(px.bar(rev, x="tenure", y="MonthlyCharges"), use_container_width=True)

    with col2:
        st.markdown("### 📊 Customer Scatter")
        st.plotly_chart(px.scatter(filtered_data, x="tenure", y="MonthlyCharges", color="Churn"), use_container_width=True)

    # ---------------- TABLES ----------------
    st.markdown("### 🔴 High Risk Customers")
    high_risk = filtered_data[filtered_data["Churn"] == 1]
    st.dataframe(high_risk.head(10), use_container_width=True)

    st.markdown("### 🏆 Top Paying Customers")
    top = filtered_data.sort_values(by="MonthlyCharges", ascending=False).head(10)
    st.dataframe(top, use_container_width=True)

    # ---------------- AI INSIGHT ----------------
    st.markdown("### 🧠 AI Insight")

    if churn_rate > 50:
        st.error("⚠️ High churn detected! Immediate action needed.")
    elif churn_rate > 30:
        st.warning("⚡ Moderate churn. Focus on retention.")
    else:
        st.success("✅ Churn is under control.")

    # ---------------- QUICK ACTIONS ----------------
    st.markdown("### ⚡ Quick Actions")

    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("📩 Send Offer"):
            st.success("Offers sent to high-risk customers!")

    with c2:
        if st.button("📞 Contact Customers"):
            st.info("Connecting to customers...")

    with c3:
        report = filtered_data.to_csv(index=False)
        st.download_button("⬇ Download Report", report, "dashboard_report.csv")
# ---------------- PREDICTION ----------------
elif page == "🤖 Prediction":

    st.title("🤖 Churn Prediction")

    col1, col2 = st.columns(2)

    with col1:
        t = st.slider("Tenure",1,72,12)
        m = st.slider("Monthly Charges",20,120,50)

    with col2:
        tot = st.number_input("Total Charges",500)
        c = st.selectbox("Contract",["Month-to-month","One year","Two year"])

    mp = {"Month-to-month":0,"One year":1,"Two year":2}

    if st.button("🚀 Predict"):

        prob = clf.predict_proba([[t,m,tot,mp[c]]])[0][1]*100

        st.metric("Churn Probability", f"{prob:.2f}%")

        if prob > 70:
            st.error("🔴 High Risk Customer")
        elif prob > 40:
            st.warning("🟡 Medium Risk")
        else:
            st.success("🟢 Low Risk")

        st.progress(int(prob))

        st.subheader("💡 Recommendation")

        if prob > 70:
            st.error("Offer discount + call immediately")
        elif prob > 40:
            st.warning("Send personalized offer")
        else:
            st.success("Customer safe")

    # Feature Importance
    st.subheader("🧠 Feature Importance")

    imp = pd.DataFrame({
        "Feature": X.columns,
        "Importance": clf.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(imp.set_index("Feature"))

    # Bulk Prediction
    st.subheader("📂 Bulk Prediction")

    file = st.file_uploader("Upload CSV")

    if file:
        df_new = pd.read_csv(file)
        preds = clf.predict(df_new)
        df_new["Prediction"] = preds
        st.dataframe(df_new)
# ---------------- CUSTOMER PROFILE ----------------
elif page == "👤 Customer Profile":

    st.title("👤 Customer Profile")

    idx = st.number_input("Customer ID", 0, len(data)-1, 0)
    cust = data.iloc[idx]

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Tenure", cust["tenure"])
        st.metric("Monthly Charges", f"₹{cust['MonthlyCharges']}")

    with col2:
        st.metric("Total Charges", f"₹{cust['TotalCharges']}")
        st.metric("Contract", cust["Contract"])

    # Prediction
    prob = clf.predict_proba([cust.drop("Churn")])[0][1]*100

    st.subheader("🤖 Churn Prediction")
    st.metric("Probability", f"{prob:.2f}%")

    # Segment
    st.subheader("📊 Segment")

    if cust["tenure"] < 12:
        st.info("New Customer")
    elif cust["tenure"] < 36:
        st.info("Mid Customer")
    else:
        st.success("Loyal Customer")

    # LTV
    st.subheader("💰 Lifetime Value")
    ltv = cust["MonthlyCharges"] * cust["tenure"]
    st.metric("LTV", f"₹{ltv:.0f}")

    # Visual
    st.subheader("📈 Position")

    fig = px.scatter(data, x="tenure", y="MonthlyCharges", color="Churn")
    fig.add_scatter(x=[cust["tenure"]], y=[cust["MonthlyCharges"]],
                    mode="markers", marker=dict(size=12,color="red"))
    st.plotly_chart(fig, use_container_width=True)

    # Recommendation
    st.subheader("💡 Recommendation")

    if prob > 70:
        st.error("Call immediately + offer discount")
    elif prob > 40:
        st.warning("Send offers")
    else:
        st.success("Safe customer")

    st.dataframe(cust)

# ---------------- ANALYTICS ----------------
elif page == "📊 Analytics":

    st.title("📊 Advanced Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Distribution")
        st.plotly_chart(px.pie(data, names="Churn", hole=0.5), use_container_width=True)

    with col2:
        st.subheader("Contract Impact")
        c = data.groupby("Contract")["Churn"].mean().reset_index()
        st.plotly_chart(px.bar(c, x="Contract", y="Churn"), use_container_width=True)

    st.subheader("Charges vs Churn")
    st.plotly_chart(px.box(data, x="Churn", y="MonthlyCharges"), use_container_width=True)

    st.subheader("Tenure Histogram")
    st.plotly_chart(px.histogram(data, x="tenure", color="Churn"), use_container_width=True)

    st.subheader("Correlation Heatmap")

    import seaborn as sns
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("Revenue at Risk")
    risk = data[data["Churn"]==1]["MonthlyCharges"].sum()
    st.metric("₹ Risk", f"{risk:.0f}")
    
# ---------------- CUSTOMER REPORT ----------------
elif page == "📄 Customer Report":

    st.title("📄 Customer Report")

    c1,c2,c3 = st.columns(3)

    c1.metric("Customers", len(data))
    c2.metric("Churn %", f"{data['Churn'].mean()*100:.2f}%")
    c3.metric("Revenue Risk", f"₹{data[data['Churn']==1]['MonthlyCharges'].sum():.0f}")

    st.subheader("Full Data")
    st.dataframe(data)

    st.subheader("High Risk Customers")
    st.dataframe(data[data["Churn"]==1].head(10))

    st.subheader("Insights")

    st.write("• Focus on high paying churn customers")
    st.write("• Improve plans for month-to-month users")
    st.write("• Retain new customers")

    st.subheader("Download")

    st.download_button(
        "⬇ Download CSV",
        data.to_csv(index=False),
        "report.csv"
    )

# ---------------- LOGOUT ----------------
elif page == "🔐 Logout":

    st.title("🔐 Logout")

    st.warning("Are you sure you want to logout?")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Yes Logout"):
            for key in st.session_state.keys():
                del st.session_state[key]

            st.success("Logged out successfully")
            st.rerun()

    with col2:
        if st.button("❌ Cancel"):
            st.info("Logout cancelled")