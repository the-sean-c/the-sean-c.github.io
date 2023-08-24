import plotly.graph_objects as go

the_template = go.layout.Template()

the_template.layout = go.Layout(
    # Transparent background
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    # Adjusting margins
    # margin=dict(t=15, b=15, l=15, r=15),
    # Updating the X and Y axis properties
    xaxis=dict(
        showline=False,  # Hide axis line
        showticklabels=True,  # Ensure that tick labels are shown
    ),
    yaxis=dict(
        showline=False,  # Hide axis line
        showticklabels=True,  # Ensure that tick labels are shown
    ),
)
