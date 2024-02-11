import streamlit as st
import pandas as pd
import plotly.express as px
from nltk.tokenize import sent_tokenize

# Set the page icon
st.set_page_config(page_title="Election News Dashboard", 
                   page_icon=":bar_chart:",
                   initial_sidebar_state="expanded")

# Load the dataframe
@st.cache_data
def load_data(file):
    data = pd.read_csv(file, encoding='utf8')
    
    # Convert the 'Date' column to datetime format
    data['Date'] = pd.to_datetime(data['Date'])
    
    return data
# Load the data

df = load_data("indonesia-election-2024-dataset.csv")

# Adjust column widths
column_widths = {
    'Date': 50,
    'Title': 200,
    'Text': 300,
    'Publication': 50
}

# Calculate total width
total_width = sum(column_widths.values())

# Set the page title
st.title("Indonesia Election 2024")

# Sidebar menu
option = st.sidebar.selectbox("Select a feature", ["Data Visualisation", "Search News", "Key Word in Context"])

if option == "Data Visualisation":
    import plotly.express as px

    # Define a custom color scale for each publication
    color_scale = px.colors.qualitative.Set2

    # Data Visualisation page
    st.header("Data Visualisation")

    st.text("The data ranges from 2023-11-29 to 2024-02-06 and includes all five\npresidential debates organized by the General Elections Commission.\nThe data sources comprise detik, kompas, and liputan6.")
    
    # Article Count per Publication
    st.subheader("Article Count per Publication")

    st.text("The length of each bar represents the number of articles, and\neach bar is color-coded to represent a different publication.")

    chart_data_total = df.groupby('Publication').size().reset_index(name='Total Article')
    fig = px.bar(chart_data_total, x='Total Article', y='Publication', orientation='h', color='Publication', color_discrete_sequence=color_scale)
    fig.update_layout(width=700, height=350, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Daily and Weekly Average Charts for each Publication
    st.subheader("Average Daily and Weekly Articles")

    st.text("These charts shows the daily and weekly averages of articles published.")

    # Calculate daily and weekly averages for each publication
    daily_avg_per_pub = df.groupby(['Date', 'Publication']).size().groupby('Publication').mean()
    weekly_avg_per_pub = df.groupby([pd.Grouper(key='Date', freq='W'), 'Publication']).size().groupby('Publication').mean()

    # Overall daily and weekly averages
    overall_daily_avg = df.groupby('Date').size().mean()
    overall_weekly_avg = df.groupby(pd.Grouper(key='Date', freq='W')).size().mean()

    # Create bar chart for daily and weekly averages for each publication
    avg_data_per_pub = pd.DataFrame({'Daily Average': daily_avg_per_pub, 'Weekly Average': weekly_avg_per_pub})
    avg_data_per_pub.reset_index(inplace=True)

    # Add overall averages to the DataFrame
    overall_avg_data = pd.DataFrame({'Publication': ['Overall'], 
                                    'Daily Average': [overall_daily_avg], 
                                    'Weekly Average': [overall_weekly_avg]})
    avg_data_per_pub = pd.concat([avg_data_per_pub, overall_avg_data])

    fig = px.bar(avg_data_per_pub, x='Publication', y=['Daily Average', 'Weekly Average'], 
                barmode='group', color_discrete_sequence=color_scale,
                labels={'Publication': 'Publication', 'value': 'Average Articles', 'variable': 'Timeframe'})
    fig.update_layout(width=700, height=350)
    st.plotly_chart(fig, use_container_width=True)

    # Article Trends Over Time
    st.subheader("Article Trends Over Time #1")

    st.text("The chart shows the trends in the number of articles published over time.")

    chart_data_all = df.groupby(['Date', 'Publication']).size().reset_index(name='Article Count')
    fig = px.line(chart_data_all, x='Date', y='Article Count', color='Publication', color_discrete_sequence=color_scale)

    # Make the line smooth
    fig.update_traces(line_shape='spline')
    fig.update_layout(width=700, height=350)
    st.plotly_chart(fig, use_container_width=True)

    # Pivot the data to prepare for a stacked bar chart
    pivot_data = chart_data_all.pivot(index='Date', columns='Publication', values='Article Count').fillna(0)

    # Function to count total article frequency over time
    def count_total_article_frequency_over_time(data: pd.DataFrame) -> pd.DataFrame:
        total_article_frequency_over_time = data.groupby('Date').size().reset_index(name='Total Articles')
        return total_article_frequency_over_time

    # Calculate total article frequency over time
    total_article_freq_over_time_df = count_total_article_frequency_over_time(df)

    # Display the line chart for total article trends over time
    st.subheader("Article Trends Over Time #2")

    st.text("This chart illustrates the total number of articles published over time\nacross all publications.")

    # Plotting the line chart for total article trends over time
    fig = px.line(total_article_freq_over_time_df, x='Date', y='Total Articles', 
                labels={'Date': 'Date', 'Total Articles': 'Total Articles'})

    # Make the line smooth
    fig.update_traces(line_shape='spline')
    fig.update_layout(width=700, height=350)
    st.plotly_chart(fig, use_container_width=True)

    # Function to count word frequency in titles over time for each candidate
    def count_word_frequency_over_time(data: pd.DataFrame, words: list) -> pd.DataFrame:
        word_frequency_over_time = {word: [] for word in words}

        for word in words:
            word_frequency_series = data[data['Title'].str.contains(word, case=False)]['Date'].value_counts().sort_index()
            word_frequency_over_time[word] = word_frequency_series

        return pd.DataFrame(word_frequency_over_time)

    # Words to count frequency for
    candidates_to_count = ['Anies', 'Muhaimin', 'Amin', 'Prabowo', 'Gibran', 'Ganjar', 'Mahfud']

    # Count word frequency over time
    word_freq_over_time_df = count_word_frequency_over_time(df, candidates_to_count)

    # Reset index to have 'Date' as a column
    word_freq_over_time_df = word_freq_over_time_df.reset_index().rename(columns={'index': 'Date'})

    # Melt the dataframe to have 'Candidate' and 'Frequency' columns
    word_freq_over_time_melted = word_freq_over_time_df.melt(id_vars='Date', var_name='Candidate', value_name='Frequency')

    # Create a line chart for candidate occurrences over time
    fig = px.line(word_freq_over_time_melted, x='Date', y='Frequency', color='Candidate', 
                color_discrete_sequence=color_scale)

    # Displaying the historical chart for candidate mentions in titles over time
    st.subheader("Candidate Mentions in Titles Over Time")

    st.text("This chart illustrates the frequency of each candidate's name\nmentioned in the news titles over time.")

    # Plotting the line chart
    fig = px.line(word_freq_over_time_melted, x='Date', y='Frequency', color='Candidate', 
                color_discrete_sequence=color_scale)

    # Add title and axis labels
    fig.update_layout(xaxis_title='Date',
                    yaxis_title='Frequency')

    # Make the line smooth
    fig.update_traces(line_shape='spline')

    # Show the chart
    st.plotly_chart(fig, use_container_width=True)

    # Function to count word frequency in titles
    def count_word_frequency_in_titles(data: pd.DataFrame, words: list) -> pd.DataFrame:
        word_frequency = {word: 0 for word in words}

        for title in data['Title']:
            for word in words:
                # Check for the word Muhaimin and Imin
                if word.lower() in title.lower() or (word.lower() == 'muhaimin' and 'imin' in title.lower()):
                    word_frequency[word] += 1

        return pd.DataFrame(list(word_frequency.items()), columns=['Word', 'Frequency'])

    # Words to count frequency for
    words_to_count = ['Anies', 'Muhaimin', 'Amin', 'Prabowo', 'Gibran', 'Ganjar', 'Mahfud']

    # Display the bar chart
    st.subheader("Candidate Mentions in Titles")

    st.text("Anies-Muhaimin is the only candidate pair with an official tagline called 'Amin'.\nAs a result, there are articles where the tagline is used interchangeably with\ntheir names, requiring separate treatment in visualising the data. Additionally,\nMuhaimin is sometimes referred to as 'Imin', 'Gus Imin' or 'Cak Imin', which\nmust be accounted for in the visualisation.")

    word_freq_df = count_word_frequency_in_titles(df, words_to_count)

    # Update the label for 'Muhaimin' to 'Muhamain' in the DataFrame
    word_freq_df.loc[word_freq_df['Word'] == 'Muhaimin', 'Word'] = 'Muhaimin'

    fig = px.bar(word_freq_df, x='Frequency', y='Word', orientation='h', color='Word', 
                color_discrete_sequence=color_scale)
    fig.update_layout(width=700, height=350, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Function to count word frequency in titles for each publication
    def count_word_frequency_in_titles_per_publication(data: pd.DataFrame, words: list) -> pd.DataFrame:
        word_frequency_per_pub = {pub: {word: 0 for word in words} for pub in data['Publication'].unique()}

        for pub in data['Publication'].unique():
            pub_data = data[data['Publication'] == pub]
            for title in pub_data['Title']:
                for word in words:
                    if word.lower() in title.lower() or (word.lower() == 'muhaimin' and 'imin' in title.lower()):
                        word_frequency_per_pub[pub][word] += 1

        return pd.DataFrame(word_frequency_per_pub).transpose()

    # Words to count frequency for
    words_to_count = ['Anies', 'Muhaimin', 'Amin', 'Prabowo', 'Gibran', 'Ganjar', 'Mahfud']

    # Display the bar chart for candidate mentions in titles per publication
    st.subheader("Candidate Mentions in Titles per Publication")

    st.text("This chart illustrates the frequency of each candidate's name\nmentioned in the news titles, categorized by publication.")

    word_freq_per_pub_df = count_word_frequency_in_titles_per_publication(df, words_to_count)

    # Melt the dataframe to have 'Publication', 'Candidate', and 'Frequency' columns
    word_freq_per_pub_melted = word_freq_per_pub_df.reset_index().melt(id_vars='index', var_name='Candidate', value_name='Frequency')
    word_freq_per_pub_melted.rename(columns={'index': 'Publication'}, inplace=True)

    # Create a horizontal stacked bar chart for candidate mentions in titles per publication
    fig = px.bar(word_freq_per_pub_melted, x='Candidate', y='Frequency', color='Publication',
                color_discrete_sequence=color_scale)

    # Add title and axis labels
    fig.update_layout(xaxis_title="Frequency",
                    yaxis_title="Candidate")

    # Show the chart
    st.plotly_chart(fig, use_container_width=True)

elif option == "Search News":

    # Search function
    st.header("Search News")
    st.text("Enter a keyword or a phrase to search for relevant news articles.")

    search_query = st.text_input("Keyword to search for")

    if search_query:
        # Filter the dataframe based on the search query
        filtered_df = df[df['Text'].str.contains(search_query, case=False)]

        # Display the search results
        st.write(f"### Search Results for \"{search_query}\"")
        st.dataframe(filtered_df[['Date', 'Title', 'Text', 'Publication']], width=total_width, height=None, use_container_width=True)

elif option == "Key Word in Context":
    # Concordance / Key Word in Context function
    def display_concordance(data: pd.DataFrame, col: str, keyword: str, window_size: int = 7) -> pd.DataFrame:
        concordance_lines = []

        for index, row in data.iterrows():
            text = row[col]
            words = text.split()
            
            for i, word in enumerate(words):
                if word == keyword:
                    start = max(0, i - window_size)
                    end = min(len(words), i + window_size + 1)
                    left_context = ' '.join(words[start:i])
                    keyword_in_context = ' '.join(words[i:i+1])
                    right_context = ' '.join(words[i+1:end])
                    
                    concordance_lines.append({
                        "Left context": left_context,
                        "Key Word": keyword_in_context,
                        "Right context": right_context,
                        "Publication": row["Publication"],
                        "Title": row["Title"]

                    })
        return pd.DataFrame(concordance_lines)

    # Concordance page
    st.header("Key Word in Context")
    st.text("Explore occurrences of a keyword within the data along with contextual snippets.")
    selected_column = st.selectbox("To use this feature, please choose either the 'Text' or 'Title' column", df.columns)
    keyword = st.text_input("Enter a keyword (only a single word)")

    if keyword:
        concordance_df = display_concordance(df, selected_column, keyword)
        st.write(f"### Key Word in Context for \"{keyword}\" in {selected_column}")
        st.dataframe(concordance_df, width=total_width, height=None, use_container_width=True)
