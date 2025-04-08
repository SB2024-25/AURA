

# import os
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# from flask import Flask, render_template, request, session, redirect, url_for
# from mistralai.client import MistralClient
# import json
# from plotly.utils import PlotlyJSONEncoder

# app = Flask(__name__, template_folder="templates")
# app.secret_key = 'f16c3fdde646a6d8ab543cb4aef67765'

# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# MISTRAL_API_KEY = "uCNyCL79rSlRcAZsDeULmQ4AziGYc6BE"
# mistral = MistralClient(api_key=MISTRAL_API_KEY)


# def preprocess_dataframe(df):
#     for col in df.columns:
#         df[col] = pd.to_numeric(df[col], errors='coerce')
#     df.dropna(axis=1, how='all', inplace=True)
#     return df


# def generate_conclusion(df, viz_type, final_summary=False):
#     try:
#         if final_summary:
#             prompt = f"""
#             You are analyzing a dataset. Please describe the dataset in a way that is easily understandable.
#             - Explain what the dataset appears to be about.
#             - Mention key trends, patterns, or insights without mathematical terms.
#             - Describe what is happening overall in **5-10 lines** in a paragraph format.
#             - Make it **human-friendly**, as if explaining to a non-technical person.
#             Dataset Overview: {df.head(3).to_string()}
#             Column Names: {list(df.columns)}
#             """
#         else:
#             prompt = f"""
#             Analyze this {viz_type} visualization and provide 6-7 key insights in bullet points.
#             Dataset Summary: {df.describe()}
#             """

#         response = mistral.chat(
#             model="mistral-small",
#             messages=[
#                 {"role": "system", "content": "You are a data analysis expert."},
#                 {"role": "user", "content": prompt},
#             ]
#         )
#         ai_output = response.choices[0].message.content
#         return ai_output if final_summary else ai_output.split("\n")[:7]
#     except Exception as e:
#         return [f"❌ Error generating insights: {e}"]


# @app.route('/')
# def index():
#     return render_template('index.html', datasets=os.listdir(UPLOAD_FOLDER))


# @app.route('/register')
# def register():
#     return render_template('register.html')


# @app.route('/middle.html')
# def middle():
#     datasets = os.listdir(UPLOAD_FOLDER)
#     return render_template('middle.html', datasets=datasets)


# @app.route('/visualization')
# def visualization():
#     return render_template('visualization.html')


# @app.route('/upload', methods=['POST'])
# def upload():
#     session["conclusions"] = {}
#     file = request.files.get('file')
#     if file and file.filename:
#         filename = file.filename
#         filepath = os.path.join(UPLOAD_FOLDER, filename)
#         file.save(filepath)
#         session["selected_dataset"] = filename
#         return redirect(url_for('middle'))
#     return redirect(url_for('index'))


# # ---------- Chart Generators with Plotly (Dynamic) ---------- #

# def generate_heatmap(df):
#     fig = px.imshow(df.corr(), text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
#     #return json.dumps(fig, cls=PlotlyJSONEncoder), generate_conclusion(df, "Heatmap")
#     return fig, generate_conclusion(df, "Heatmap")


# def generate_bar_chart(df):
#     numeric_cols = df.select_dtypes(include='number')
#     if numeric_cols.empty:
#         return None, ["No numeric data available for bar chart."]
    
#     data = numeric_cols.mean().sort_values(ascending=False)
#     fig = px.bar(x=data.index, y=data.values, labels={'x': 'Feature', 'y': 'Average'}, title="Bar Chart")

#     #return json.dumps(fig, cls=PlotlyJSONEncoder), generate_conclusion(df, "Bar Chart")
#     return fig, generate_conclusion(df, "bar chart")


# def generate_area_chart(df):
#     numeric_cols = df.select_dtypes(include='number').columns
#     if numeric_cols.empty:
#         return None, ["No numeric data for area chart."]
    
#     fig = px.area(df, x=df.index, y=numeric_cols, title="Area Chart")

#     #return json.dumps(fig, cls=PlotlyJSONEncoder), generate_conclusion(df, "Area Chart")
#     return fig, generate_conclusion(df, "Area Chart")

# def generate_scatter_plot(df):
#     numeric_cols = df.select_dtypes(include='number').columns
#     if len(numeric_cols) < 2:
#         return None, ["Not enough numeric columns for scatter plot."]

#     fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], title="Scatter Plot")

#     #return json.dumps(fig, cls=PlotlyJSONEncoder), generate_conclusion(df, "Scatter Plot")
#     return fig, generate_conclusion(df, "scatter plot")


# def generate_bubble_chart(df):
#     if df.shape[1] >= 3:
#         fig = px.scatter(df, x=df.columns[0], y=df.columns[1], size=df.columns[2], opacity=0.5, title="Bubble Chart")
#     #return json.dumps(fig, cls=PlotlyJSONEncoder), generate_conclusion(df, "Bubble Chart")
#     return fig, generate_conclusion(df, "bubble chart")
#     return None, "Insufficient columns for bubble chart."

# def generate_pie_chart(df):
#     if df.empty or not isinstance(df, pd.DataFrame):
#         return None, ["No data for pie chart."]

#     col = df.select_dtypes(include='object').columns
#     if col.empty:
#         return None, ["No categorical column for pie chart."]
    
#     counts = df[col[0]].value_counts().nlargest(10)
#     fig = px.pie(names=counts.index, values=counts.values, title="Pie Chart")

#     #return json.dumps(fig, cls=PlotlyJSONEncoder), generate_conclusion(df, "Pie Chart")
#     return fig, generate_conclusion(df, "Pie Chart") 



# def generate_line_chart(df):
#     numeric_cols = df.select_dtypes(include='number').columns

#     if numeric_cols.empty:
#         return None, ["No numeric columns available for line chart."]

#     fig = px.line(df, x=df.index, y=numeric_cols, title="Line Chart")

#     #return json.dumps(fig, cls=PlotlyJSONEncoder), generate_conclusion(df, "Line Chart")
#     return fig, generate_conclusion(df, "Line chart")

# def generate_box_plot(df):
#     numeric_cols = df.select_dtypes(include='number').columns
#     if numeric_cols.empty:
#         return None, ["No numeric columns available for box plot."]

#     melted_df = df[numeric_cols].melt(var_name='Feature', value_name='Value')
#     fig = px.box(melted_df, x='Feature', y='Value', title="Box Plot")

#     #return json.dumps(fig, cls=PlotlyJSONEncoder), generate_conclusion(df, "Box Plot")
#     return fig, generate_conclusion(df, "Box Plot")



# def generate_radar_chart(df):
#     numeric_cols = df.select_dtypes(include='number').columns
#     if len(numeric_cols) < 3:
#         return None, ["Need at least 3 numeric columns for radar chart."]

#     mean_values = df[numeric_cols].mean()
#     fig = go.Figure()
#     fig.add_trace(go.Scatterpolar(
#         r=mean_values.values,
#         theta=mean_values.index,
#         fill='toself',
#         name='Feature Averages'
#     ))

#     fig.update_layout(
#         polar=dict(radialaxis=dict(visible=True)),
#         showlegend=False,
#         title="Radar Chart"
#     )

#     #return json.dumps(fig, cls=PlotlyJSONEncoder), generate_conclusion(df, "Radar Chart")
#     return fig, generate_conclusion(df, "Radar Chart")



# from flask import render_template, request, redirect, url_for
# import pandas as pd
# import os
# import json
# from plotly.utils import PlotlyJSONEncoder

# @app.route('/analyze', methods=['POST'])
# def analyze():
#     selected_file = request.form.get('dataset')
#     if not selected_file:
#         return redirect(url_for('index'))

#     df = pd.read_csv(os.path.join(UPLOAD_FOLDER, selected_file))
#     numeric_df = preprocess_dataframe(df.copy())

#     conclusions = {}
#     image_dict = {}

#     if not numeric_df.empty:
#         # Each generate_* function should return a Plotly figure object
#         fig, conclusions["heatmap"] = generate_heatmap(numeric_df)
#         image_dict["heatmap"] = json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

#         fig, conclusions["bar_chart"] = generate_bar_chart(df)
#         image_dict["bar_chart"] = json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

#         fig, conclusions["pie_chart"] = generate_pie_chart(df)
#         image_dict["pie_chart"] = json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

#         fig, conclusions["scatter_plot"] = generate_scatter_plot(numeric_df)
#         image_dict["scatter_plot"] = json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

#         fig, conclusions["bubble_chart"] = generate_bubble_chart(numeric_df)
#         image_dict["bubble_chart"] = json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

#         fig, conclusions["area_chart"] = generate_area_chart(numeric_df)
#         image_dict["area_chart"] = json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

#         fig, conclusions["line_chart"] = generate_line_chart(numeric_df)
#         image_dict["line_chart"] = json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

#         fig, conclusions["box_plot"] = generate_box_plot(numeric_df)
#         image_dict["box_plot"] = json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

#         fig, conclusions["radar_chart"] = generate_radar_chart(numeric_df)
#         image_dict["radar_chart"] = json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

#     final_conclusion = generate_conclusion(numeric_df, None, final_summary=True)

#     return render_template(
#         "visualization.html",
#         conclusions=conclusions,
#         final_conclusion=final_conclusion,
#         images=image_dict
#     )



# if __name__ == '__main__':
#     app.run(debug=True)


# import os
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# from flask import Flask, render_template, request, session, redirect, url_for
# from mistralai.client import MistralClient
# import json
# from plotly.utils import PlotlyJSONEncoder
# import matplotlib.pyplot as plt

# app = Flask(__name__, template_folder="templates")
# app.secret_key = 'f16c3fdde646a6d8ab543cb4aef67765'

# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# MISTRAL_API_KEY = "uCNyCL79rSlRcAZsDeULmQ4AziGYc6BE"
# mistral = MistralClient(api_key=MISTRAL_API_KEY)


# def preprocess_dataframe(df):
#     # Remove all non-numeric columns for heatmap generation
#     df = df.select_dtypes(include='number')
    
#     if df.empty:
#         raise ValueError("No numeric data available.")

#     # Try converting columns to numeric
#     for col in df.columns:
#         df[col] = pd.to_numeric(df[col], errors='coerce')

#     # Drop columns that do not have any numeric values
#     df.dropna(axis=1, how='all', inplace=True)

#     return df


# def generate_conclusion(df, viz_type, final_summary=False):
#     try:
#         if final_summary:
#             prompt = f"""
#             You are analyzing a dataset. Please describe the dataset in a way that is easily understandable.
#             - Explain what the dataset appears to be about.
#             - Mention key trends, patterns, or insights without mathematical terms.
#             - Describe what is happening overall in **5-10 lines** in a paragraph format.
#             - Make it **human-friendly**, as if explaining to a non-technical person.
#             Dataset Overview: {df.head(3).to_string()}
#             Column Names: {list(df.columns)}
#             """
#         else:
#             prompt = f"""
#             Analyze this {viz_type} visualization and provide 6-7 key insights in bullet points.
#             Dataset Summary: {df.describe()}
#             """

#         response = mistral.chat(
#             model="mistral-small",
#             messages=[
#                 {"role": "system", "content": "You are a data analysis expert."},
#                 {"role": "user", "content": prompt},
#             ]
#         )
#         ai_output = response.choices[0].message.content
#         return ai_output if final_summary else ai_output.split("\n")[:7]
#     except Exception as e:
#         return [f"❌ Error generating insights: {e}"]


# @app.route('/')
# def index():
#     return render_template('index.html', datasets=os.listdir(UPLOAD_FOLDER))


# @app.route('/register')
# def register():
#     return render_template('register.html')


# @app.route('/middle.html')
# def middle():
#     datasets = os.listdir(UPLOAD_FOLDER)
#     return render_template('middle.html', datasets=datasets)


# @app.route('/visualization')
# def visualization():
#     return render_template('visualization.html')


# @app.route('/upload', methods=['POST'])
# def upload():
#     session["conclusions"] = {}
#     file = request.files.get('file')
#     if file and file.filename:
#         filename = file.filename
#         filepath = os.path.join(UPLOAD_FOLDER, filename)
#         file.save(filepath)
#         session["selected_dataset"] = filename
#         return redirect(url_for('middle'))
#     return redirect(url_for('index'))


# # ---------- Chart Generators with Plotly (Dynamic) ---------- #

# import plotly.express as px
# import pandas as pd

# def generate_heatmap(df):
#     # Filter for numeric columns
#     numeric_df = df.select_dtypes(include='number')

#     # Check if there are any numeric columns
#     if numeric_df.empty:
#         return None, ["No numeric data available for heatmap."]

#     # Fill missing values with zeros
#     numeric_df.fillna(0, inplace=True)

#     # Create the correlation matrix
#     correlation_matrix = numeric_df.corr()

#     # Check if the correlation matrix is empty or not square
#     if correlation_matrix.shape[0] == 0 or correlation_matrix.shape[1] == 0:
#         return None, ["Correlation matrix is empty."]

#     # Debugging: Print the correlation matrix
#     print("Correlation Matrix:")
#     print(correlation_matrix)

#     # Check if the correlation matrix has meaningful values
#     if correlation_matrix.isnull().all().all():
#         return None, ["Correlation matrix contains only NaN values."]

#     # Generate heatmap
#     fig = px.imshow(correlation_matrix, 
#                     text_auto=True, 
#                     aspect="auto", 
#                     color_continuous_scale="RdBu_r",
#                     title="Correlation Heatmap")
    
#     return fig, generate_conclusion(df, "Heatmap")

#     # Create the correlation matrix
#     #correlation_matrix = numeric_df.corr()

#     # Verify the shape of the correlation matrix
#     #if correlation_matrix.shape[0] == 0 or correlation_matrix.shape[1] == 0:
#         #return None, ["Correlation matrix is empty."]

#     # Generate heatmap
#     #fig = px.imshow(correlation_matrix, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
    
#     #return fig, generate_conclusion(df, "Heatmap")


# def generate_bar_chart(df):
#     numeric_cols = df.select_dtypes(include='number')
#     if numeric_cols.empty:
#         return None, ["No numeric data available for bar chart."]
    
#     data = numeric_cols.mean().sort_values(ascending=False)
#     fig = px.bar(x=data.index, y=data.values, labels={'x': 'Feature', 'y': 'Average'}, title="Bar Chart")
    
#     return fig, generate_conclusion(df, "Bar Chart")


# def generate_area_chart(df):
#     numeric_cols = df.select_dtypes(include='number').columns
#     if numeric_cols.empty:
#         return None, ["No numeric data for area chart."]
    
#     fig = px.area(df, x=df.index, y=numeric_cols, title="Area Chart")
    
#     return fig, generate_conclusion(df, "Area Chart")


# def generate_scatter_plot(df):
#     numeric_cols = df.select_dtypes(include='number').columns
#     if len(numeric_cols) < 2:
#         return None, ["Not enough numeric columns for scatter plot."]

#     fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], title="Scatter Plot")

#     return fig, generate_conclusion(df, "scatter plot")


# def generate_bubble_chart(df):
#     if df.shape[1] >= 3:
#         fig = px.scatter(df, x=df.columns[0], y=df.columns[1], size=df.columns[2], opacity=0.5, title="Bubble Chart")
#         return fig, generate_conclusion(df, "bubble chart")
    
#     return None, ["Insufficient columns for bubble chart."]


# def generate_pie_chart(df):
#     if df.empty or not isinstance(df, pd.DataFrame):
#         return None, ["No data for pie chart."]

#     col = df.select_dtypes(include='object').columns
#     if col.empty:
#         return None, ["No categorical column for pie chart."]
    
#     counts = df[col[0]].value_counts().nlargest(10)
#     fig = px.pie(names=counts.index, values=counts.values, title="Pie Chart")

#     return fig, generate_conclusion(df, "Pie Chart") 


# def generate_line_chart(df):
#     numeric_cols = df.select_dtypes(include='number').columns

#     if numeric_cols.empty:
#         return None, ["No numeric columns available for line chart."]

#     fig = px.line(df, x=df.index, y=numeric_cols, title="Line Chart")

#     return fig, generate_conclusion(df, "Line chart")


# def generate_box_plot(df):
#     numeric_cols = df.select_dtypes(include='number').columns
#     if numeric_cols.empty:
#         return None, ["No numeric columns available for box plot."]

#     melted_df = df[numeric_cols].melt(var_name='Feature', value_name='Value')
#     fig = px.box(melted_df, x='Feature', y='Value', title="Box Plot")

#     return fig, generate_conclusion(df, "Box Plot")


# def generate_radar_chart(df):
#     numeric_cols = df.select_dtypes(include='number').columns
#     if len(numeric_cols) < 3:
#         return None, ["Need at least 3 numeric columns for radar chart."]

#     mean_values = df[numeric_cols].mean()
#     fig = go.Figure()
#     fig.add_trace(go.Scatterpolar(
#         r=mean_values.values,
#         theta=mean_values.index,
#         fill='toself',
#         name='Feature Averages'
#     ))

#     fig.update_layout(
#         polar=dict(radialaxis=dict(visible=True)),
#         showlegend=False,
#         title="Radar Chart"
#     )

#     return fig, generate_conclusion(df, "Radar Chart")


# @app.route('/analyze', methods=['POST'])
# def analyze():
#     selected_file = request.form.get('dataset')
#     if not selected_file:
#         return redirect(url_for('index'))

#     df = pd.read_csv(os.path.join(UPLOAD_FOLDER, selected_file))
#     numeric_df = preprocess_dataframe(df.copy())

#     conclusions = {}
#     image_dict = {}

#     if not numeric_df.empty:
#         visualizations = [
#             ("heatmap", generate_heatmap),
#             ("bar_chart", generate_bar_chart),
#             ("pie_chart", generate_pie_chart),
#             ("scatter_plot", generate_scatter_plot),
#             ("bubble_chart", generate_bubble_chart),
#             ("area_chart", generate_area_chart),
#             ("line_chart", generate_line_chart),
#             ("box_plot", generate_box_plot),
#             ("radar_chart", generate_radar_chart)
#         ]

#         for chart_name, chart_func in visualizations:
#             fig, conclusions[chart_name] = chart_func(numeric_df if 'scatter' in chart_name or 'bubble' in chart_name or 'area' in chart_name or 'line' in chart_name or 'box' in chart_name or 'radar' in chart_name else df)
#             image_dict[chart_name] = json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

#     final_conclusion = generate_conclusion(numeric_df, None, final_summary=True)

#     return render_template(
#         "visualization.html",
#         conclusions=conclusions,
#         final_conclusion=final_conclusion,
#         images=image_dict
#     )


# if __name__ == '__main__':
#     app.run(debug=True)



import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, render_template, request, session, redirect, url_for, send_from_directory
from mistralai.client import MistralClient
import json
from plotly.utils import PlotlyJSONEncoder
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__, template_folder="templates")
app.secret_key = 'f16c3fdde646a6d8ab543cb4aef67765'

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MISTRAL_API_KEY = "uCNyCL79rSlRcAZsDeULmQ4AziGYc6BE"
mistral = MistralClient(api_key=MISTRAL_API_KEY)

def preprocess_dataframe(df):
    # Remove all non-numeric columns for heatmap generation
    df = df.select_dtypes(include='number')
    
    if df.empty:
        raise ValueError("No numeric data available.")

    # Try converting columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop columns that do not have any numeric values
    df.dropna(axis=1, how='all', inplace=True)

    return df

def generate_conclusion(df, viz_type, final_summary=False):
    try:
        if final_summary:
            prompt = f"""
            You are analyzing a dataset. Please describe the dataset in a way that is easily understandable.
            - Explain what the dataset appears to be about.
            - Mention key trends, patterns, or insights without mathematical terms.
            - Describe what is happening overall in **5-10 lines** in a paragraph format.
            - Make it **human-friendly**, as if explaining to a non-technical person.
            Dataset Overview: {df.head(3).to_string()}
            Column Names: {list(df.columns)}
            """
        else:
            prompt = f"""
            Analyze this {viz_type} visualization and provide 6-7 key insights in bullet points.
            Dataset Summary: {df.describe()}
            """

        response = mistral.chat(
            model="mistral-small",
            messages=[
                {"role": "system", "content": "You are a data analysis expert."},
                {"role": "user", "content": prompt},
            ]
        )
        ai_output = response.choices[0].message.content
        return ai_output if final_summary else ai_output.split("\n")[:7]
    except Exception as e:
        return [f"❌ Error generating insights: {e}"]

def generate_heatmap(df, filename):
    numeric_df = df.select_dtypes(include='number')

    if numeric_df.shape[1] < 2:
        return None, ["❌ Not enough numeric columns to generate a heatmap. At least 2 required."]

    plt.figure(figsize=(6, 6), dpi=300)  # High-res
    sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', square=True,
                cbar=True, linewidths=0.5, linecolor='white')

    plt.title("Correlation Heatmap", fontsize=16)
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close('all')

    return filename, generate_conclusion(df, "Heatmap")


@app.route('/')
def index():
    return render_template('index.html', datasets=os.listdir(UPLOAD_FOLDER))

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/middle.html')
def middle():
    datasets = os.listdir(UPLOAD_FOLDER)
    return render_template('middle.html', datasets=datasets)

@app.route('/visualization')
def visualization():
    return render_template('visualization.html')

@app.route('/upload', methods=['POST'])
def upload():
    session["conclusions"] = {}
    file = request.files.get('file')
    if file and file.filename:
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        session["selected_dataset"] = filename
        return redirect(url_for('middle'))
    return redirect(url_for('index'))

# ---------- Chart Generators with Plotly (Dynamic) ---------- #



# def generate_bar_chart(df):
#     fig = px.bar(df, x=df.columns[0], y=df.columns[1])
#     return fig, generate_conclusion(df, "Bar Chart")

# def generate_pie_chart(df):
#     fig = px.pie(df, names=df.columns[0], values=df.columns[1])
#     return fig, generate_conclusion(df, "Pie Chart")

# def generate_scatter_plot(df):
#     fig = px.scatter(df, x=df.columns[0], y=df.columns[1])
#     return fig, generate_conclusion(df, "Scatter Plot")

# def generate_bubble_chart(df):
#     fig = px.scatter(df, x=df.columns[0], y=df.columns[1], size=df.columns[2])
#     return fig, generate_conclusion(df, "Bubble Chart")

# def generate_area_chart(df):
#     fig = px.area(df, x=df.columns[0], y=df.columns[1])
#     return fig, generate_conclusion(df, "Area Chart")

# def generate_line_chart(df):
#     fig = px.line(df, x=df.columns[0], y=df.columns[1])
#     return fig, generate_conclusion(df, "Line Chart")

# def generate_box_plot(df):
#     fig = px.box(df, x=df.columns[0], y=df.columns[1])
#     return fig, generate_conclusion(df, "Box Plot")

# def generate_radar_chart(df):
#     fig = go.Figure()
#     for col in df.columns[1:]:
#         fig.add_trace(go.Scatterpolar(
#             r=df[col],
#             theta=df.columns[1:],
#             fill='toself',
#             name=col
#         ))
#     return fig, generate_conclusion(df, "Radar Chart")

import plotly.express as px
import plotly.graph_objects as go

def fallback_chart(title, message):
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False, font=dict(size=20))
    fig.update_layout(title=title, template="plotly_white", width=900, height=600)
    return fig

def generate_bar_chart(df):
    if df.shape[1] >= 2:
        try:
            fig = px.bar(df, x=df.columns[0], y=df.columns[1], template="plotly_white", title="Bar Chart")
            fig.update_layout(width=900, height=600, title_font_size=20, font=dict(size=14))
            return fig, generate_conclusion(df, "Bar Chart")
        except:
            pass
    return fallback_chart("Bar Chart", "Not enough valid data"), ["❌ Bar chart could not be generated."]



def generate_pie_chart(df):
    if df.shape[1] >= 2:
        try:
            fig = px.pie(df, names=df.columns[0], values=df.columns[1], template="plotly_white", title="Pie Chart", hole=0.3)
            fig.update_layout(width=900, height=600, title_font_size=20, font=dict(size=14))
            return fig, generate_conclusion(df, "Pie Chart")
        except:
            pass
    return fallback_chart("Pie Chart", "Not enough valid data"), ["❌ Pie chart could not be generated."]

def generate_scatter_plot(df):
    if df.shape[1] >= 2:
        try:
            fig = px.scatter(df, x=df.columns[0], y=df.columns[1], template="plotly_white", title="Scatter Plot", opacity=0.8)
            fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
            fig.update_layout(width=900, height=600, title_font_size=20, font=dict(size=14))
            return fig, generate_conclusion(df, "Scatter Plot")
        except:
            pass
    return fallback_chart("Scatter Plot", "Not enough valid data"), ["❌ Scatter plot could not be generated."]

def generate_bubble_chart(df):
    if df.shape[1] >= 3:
        try:
            fig = px.scatter(df, x=df.columns[0], y=df.columns[1], size=df.columns[2], template="plotly_white", title="Bubble Chart", opacity=0.7)
            fig.update_traces(marker=dict(sizemode='area', line=dict(width=1, color='DarkSlateGrey')))
            fig.update_layout(width=900, height=600, title_font_size=20, font=dict(size=14))
            return fig, generate_conclusion(df, "Bubble Chart")
        except:
            pass
    return fallback_chart("Bubble Chart", "Not enough valid data"), ["❌ Bubble chart could not be generated."]

def generate_area_chart(df):
    if df.shape[1] >= 2:
        try:
            fig = px.area(df, x=df.columns[0], y=df.columns[1], template="plotly_white", title="Area Chart")
            fig.update_layout(width=900, height=600, title_font_size=20, font=dict(size=14))
            return fig, generate_conclusion(df, "Area Chart")
        except:
            pass
    return fallback_chart("Area Chart", "Not enough valid data"), ["❌ Area chart could not be generated."]

def generate_line_chart(df):
    if df.shape[1] >= 2:
        try:
            fig = px.line(df, x=df.columns[0], y=df.columns[1], template="plotly_white", title="Line Chart")
            fig.update_traces(line=dict(width=3))
            fig.update_layout(width=900, height=600, title_font_size=20, font=dict(size=14))
            return fig, generate_conclusion(df, "Line Chart")
        except:
            pass
    return fallback_chart("Line Chart", "Not enough valid data"), ["❌ Line chart could not be generated."]

def generate_box_plot(df):
    if df.shape[1] >= 2:
        try:
            fig = px.box(df, x=df.columns[0], y=df.columns[1], template="plotly_white", title="Box Plot")
            fig.update_layout(width=900, height=600, title_font_size=20, font=dict(size=14))
            return fig, generate_conclusion(df, "Box Plot")
        except:
            pass
    return fallback_chart("Box Plot", "Not enough valid data"), ["❌ Box plot could not be generated."]

def generate_radar_chart(df):
    if df.shape[1] >= 2:
        try:
            fig = go.Figure()
            for i in range(len(df)):
                fig.add_trace(go.Scatterpolar(
                    r=df.iloc[i, 1:].values,
                    theta=df.columns[1:],
                    fill='toself',
                    name=str(df.iloc[i, 0])
                ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                showlegend=True,
                template="plotly_white",
                title="Radar Chart",
                width=900,
                height=600,
                title_font_size=20,
                font=dict(size=14)
            )
            return fig, generate_conclusion(df, "Radar Chart")
        except:
            pass
    return fallback_chart("Radar Chart", "Not enough valid data"), ["❌ Radar chart could not be generated."]


@app.route('/analyze', methods=['POST'])
def analyze():
    selected_file = request.form.get('dataset')
    if not selected_file:
        return redirect(url_for('index'))

    df = pd.read_csv(os.path.join(UPLOAD_FOLDER, selected_file))
    numeric_df = preprocess_dataframe(df.copy())

    conclusions = {}
    image_dict = {}

     #Generate static heatmap
    heatmap_filename = os.path.join(UPLOAD_FOLDER, "heatmap.png")
    heatmap_path, conclusions["heatmap"] = generate_heatmap(numeric_df, heatmap_filename)
    image_dict["heatmap"] = heatmap_path  # Store the path to the static image

    # Generate other visualizations dynamically
    visualizations = [
        ("bar_chart", generate_bar_chart),
        ("pie_chart", generate_pie_chart),
        ("scatter_plot", generate_scatter_plot),
        ("bubble_chart", generate_bubble_chart),
        ("area_chart", generate_area_chart),
        ("line_chart", generate_line_chart),
        ("box_plot", generate_box_plot),
        ("radar_chart", generate_radar_chart)
    ]

    for chart_name, chart_func in visualizations:
        fig, conclusions[chart_name] = chart_func(numeric_df if 'scatter' in chart_name or 'bubble' in chart_name or 'area' in chart_name or 'line' in chart_name or 'box' in chart_name or 'radar' in chart_name else df)
        image_dict[chart_name] = json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

    final_conclusion = generate_conclusion(numeric_df, None, final_summary=True)

    return render_template(
        "visualization.html",
        conclusions=conclusions,
        final_conclusion=final_conclusion,
        images=image_dict
    )

@app.route('/uploads/<path:filename>')
def send_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)