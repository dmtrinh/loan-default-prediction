#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt

# Define a function to print all unique values for each column
def print_unique_values(df):
    for col in df.columns:
        # if count of unique values is less than 20, print the unique values
        if len(df[col].unique()) < 20:
            print(f'{col}: {df[col].unique()}')

# Define function to print the percentage of loan_status for each value in a column
def print_loan_status_percentage(df, column_name):
    for i in df[column_name].unique():
        if (i < 10):
            print(f'{column_name} = {i}')
            print(df[df[column_name] == i]['loan_status'].value_counts(normalize=True))
            print('----')

# Function to create a correlation matrix
def plot_corr(df, size=10):
    """
    Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot

    Displays:
        matrix of correlation between columns.  Blue-cyan-yellow-red-darkred => less to more correlated
                                                0 ------------------>  1
                                                Expect a darkred line running from top left to bottom right
    """

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    cax = ax.matshow(corr, cmap='coolwarm')
    fig.colorbar(cax)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.savefig('./output/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
def create_bins(df, col, n_bins):
    """
    Function to create bins for a given column in the dataframe

    Input:
        df: pandas DataFrame
        col: column name to create bins for
        n_bins: number of bins to create

    Returns:
        df: pandas DataFrame with a new column that contains the bin number for each row
    """

    # if col is of type 'object', attempt to convert it to a numeric type
    if df[col].dtype == 'object':
        print(f'Column {col} is of type object')
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            print(f'Column {col} is not numeric')
            try:
                # Attempt to parse percentage values
                print(f'Attempting to convert {col} to numeric')
                df[col] = df[col].str.rstrip('%').astype('float64')
            except:
                print(f'Unable to convert {col} to numeric')
                return df

    
    # Create a new column with the column name + '_bin'
    new_col = col + '_bin'
    # Create bins for the column
    df[new_col] = pd.cut(df[col], bins=n_bins)
    return df

def create_wordcloud(input: pd.Series, max_words: int = 1000):
    from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

    print(f'Number of unique values: {input.nunique()}')
    
    # replace space in values with underscore
    input = input.str.replace(' ', '_')

    # create string out of all values in input series
    text = ' '.join(input.dropna())

    wordcloud = WordCloud(
        width=1920, 
        height=1080, 
        max_words=max_words, 
        random_state=42, 
#        colormap='cividis'
#        colormap='plasma'
        colormap='cool'
    ).generate(text)

    # Generate plot
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig('./output/wordcloud_' + input.name + '.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_Atotals_vs_B(df, A, B, rotation=0):
    """
    Function to plot the total of one feature against another feature

    Input:
        df: pandas DataFrame
        feature1: name of the first feature
        feature2: name of the second feature
    """

    # Total A by B
    total_A = df.groupby(B)[A].sum().sort_values(ascending=False)

    # Plot total A by B
    plt.figure(figsize=(18, 6))
    ax = sns.barplot(x=B, y=A, hue=B, data=pd.DataFrame(total_A), palette='viridis')

    # Add space to left most bar
    #plt.xlim(-1, 7)

    # Add more margin between each bar
    plt.subplots_adjust(wspace=.75)

    # Add labels to each bar, format y-axis in millions
    #for container in ax.containers:
    #    ax.bar_label(container, fmt='${:,.2f}')

    plt.title(f'Total {A} Volume by {B}')
    # Set y-label
    plt.ylabel(f'Total {A} Volume ($)')

    # Set x-tick rotation
    plt.xticks(rotation=rotation)

    # Show y-axis in plain format (without scientific notation)
    plt.ticklabel_format(style='plain', axis='y')
    plt.savefig(f'./output/total_{A}_by_{B}.png', dpi=300, bbox_inches='tight')
    plt.show()
    

def plot_dataframe_as_table_image(df, table_name, index_title):
    """
    Function to plot a dataframe as an image

    Input:
        df: pandas DataFrame
        title: title of the plot
    """

    fig = ff.create_table(df, index=True, index_title=index_title)
    fig.update_layout(
        autosize=True,
        width=800
    )

    fig.write_image('./output/table_' + table_name + '.png', scale=2)
    fig.show()