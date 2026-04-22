import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

st.set_page_config(page_title="RUHM Demo App", layout="wide")
st.title("RUHM Streamlit Ready App")
st.markdown("This app verifies the installed libraries and shows a sample chart.")

package_versions = {
    "streamlit": st.__version__,
    "pandas": pd.__version__,
    "numpy": np.__version__,
    "matplotlib": matplotlib.__version__,
}

st.subheader("Installed Package Versions")
st.table(package_versions)

st.subheader("Sample Data and Charts")
df = pd.DataFrame({
    "x": np.arange(1, 11),
    "y": np.random.normal(loc=0.0, scale=1.0, size=10).cumsum(),
})

st.write("### Data Preview")
st.dataframe(df)

st.write("### Line chart")
st.line_chart(df.set_index("x"))

fig, ax = plt.subplots()
ax.plot(df["x"], df["y"], marker="o", linestyle="-", color="#1f77b4")
ax.set_title("Cumulative Random Series")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True)

st.write("### Matplotlib chart")
st.pyplot(fig)
