import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6.4, 2.4))
# Calculate the percentage of borrowers in each FICO segment by grade
fico_segments = df.groupby(['grade', pd.cut(df['fico'], bins=[0, 600, 650, 700, 750, 800, 850])]).size().unstack()
fico_segments_percentage = fico_segments.apply(lambda x: x / x.sum() * 100, axis=1)

# Plot the bar graph
fico_segments_percentage.plot(kind='bar', stacked=True, ax=ax)

# Add a title to the plot
ax.set_title("Percentage of Borrowers in FICO Segments by Grade")

# Label the x and y axes of the plot
ax.set_xlabel("Grade")
ax.set_ylabel("Percentage of Borrowers")

# Show the legend outside the graph
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
st.pyplot(fig)