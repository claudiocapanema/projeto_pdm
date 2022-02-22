
import numpy as np
import pandas as pd
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import isnan, when, count, col
import pyspark.sql.functions as F
import seaborn as sns
import os

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.fpm import FPGrowth

# word cloud library
from wordcloud import WordCloud
import requests
import matplotlib.pyplot as plt
import seaborn as sns
#requests.packages.urllib3.disable_warnings()
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.feature import NGram
from pyspark.sql.types import *
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from sklearn.metrics import confusion_matrix
from pyspark.sql import SparkSession

from sklearn.metrics import classification_report
import plotly.graph_objects as go

def one(df_wordcloud, df_scatter, artist_column, timestamp_column, wordcloud_filename, lineplot_filename):

    one_wordcloud(df_wordcloud, artist_column, wordcloud_filename)

    one_scatter(df_scatter, timestamp_column, lineplot_filename)

def one_scatter(df, column, filename):


    x = df['year'].unique().tolist()
    print("anos", x)
    df = df.sort_values('year')
    artists = df['artist'].unique().tolist()
    fig = go.Figure()
    colors = ["rgb(184, 247, 212)", "rgb(184, 247, 252)", "rgb(184, 200, 212)", "rgb(130, 247, 212)",
              "rgb(184, 150, 212)", "rgb(184, 247, 100)", "rgb(50, 247, 212)", "rgb(50, 147, 112)",
              "rgb(250, 247, 212)", "rgb(250, 147, 130)"]
    # import plotly.express as px
    # px.line()
    for i in range(len(artists)):
        artist = artists[i]
        df_artist = df[df['artist'] == artist]
        fig.add_trace(go.Scatter(
            x=df_artist['year'].tolist(), y=df_artist['Porcentagem'].tolist(),
            mode='lines',
            name=artist,
            line=dict(width=0.5, color=colors[i]),
            stackgroup='one',
            groupnorm='percent'  # sets the normalization for the sum of the stackgroup
        ))

    fig.update_layout(
        showlegend=True,
        xaxis_type='category',
        title_text="Porcentagem de músicas ouvidas de cada artista (dentre os top 10 artistas) no tempo",
        yaxis=dict(
            type='linear',
            range=[1, 100],
            ticksuffix='%'))

    fig.write_image(filename)

def one_wordcloud(df, column, filename):

    df = df.head(200)
    total = df['count'].sum()
    print("Top 10 artistas")
    print(df.head(10))
    df['count'] = df['count'].to_numpy() / total

    unique_artists = df[column].unique().tolist()
    artists_dict = {}
    for artist in unique_artists:

        artist_count = df[df[column] == artist]['count']
        artists_dict[artist] = float(artist_count)
    plt.figure()

    wordcloud = WordCloud(
        background_color='white',
        width=512,
        random_state=1,
        height=384
    ).generate_from_frequencies(artists_dict)
    plt.imshow(wordcloud)
    plt.tight_layout()
    plt.axis('off')
    plt.savefig(filename)

def two_bar_one(df):

    plt.figure()

    fig = sns.barplot(data=df, x='artist', y='precision')
    plt.xticks(rotation=70)
    plt.tight_layout()
    fig = fig.get_figure()

    fig.savefig("two_precision.png")

    plt.figure()

    fig = sns.barplot(data=df, x='artist', y='recall')
    plt.xticks(rotation=70)
    plt.tight_layout()
    fig = fig.get_figure()

    fig.savefig("two_recall.png")
    plt.figure()

    fig = sns.barplot(data=df, x='artist', y='f1-score')
    plt.xticks(rotation=70)
    plt.tight_layout()
    fig = fig.get_figure()
    fig.savefig("two_fscore.png")

def two_bar_two(df):

    plt.figure()
    print("dataframe")
    print(df)
    fig = sns.barplot(data=df, x='Métrica', y='Porcentagem')
    plt.xticks(rotation=70)
    plt.tight_layout()
    fig = fig.get_figure()

    fig.savefig("two_metrica.png")



# def three(df):
#
#     df = df.sort_values('created_at')
#     first = int(df.head(1)['followers_count'])
#     last = int(df.tail(1)['followers_count'])
#     user_id = str(df.head(1)['user_id'])
#     total = len(df)
#     ngrams_list = df['ngrams'].to_numpy().flatten().tolist()
#     ngrams_list = np.concatenate(ngrams_list)
#     df_ngrams = pd.DataFrame({'ngrams': ngrams_list, 'teste': ngrams_list})
#     df_ngrams = df_ngrams.groupby('ngrams').count().reset_index().sort_values('teste', ascending=False)
#     ngrams_list = df_ngrams.head(5)['ngrams'].tolist()
#     added = last - first
#
#     return pd.DataFrame({'user_id': [user_id], 'new_followers': [added], 'total_tweets': [total], 'ngrams': [str(ngrams_list)]})

def three_(df, filename):
    x = df['year'].unique().tolist()
    print("entra", df)
    df = df.head(10)
    df = df[df['year'] != 2013]
    df = df.sort_values('year')
    artists = df['country'].unique().tolist()
    fig = go.Figure()
    colors = ["rgb(184, 247, 212)",
              "rgb(184, 150, 212)", "rgb(184, 247, 100)", "rgb(50, 247, 212)", "rgb(50, 147, 112)",
              "rgb(250, 247, 212)", "rgb(250, 147, 130)"]
    # import plotly.express as px
    # px.line()
    for i in range(len(artists)):
        artist = artists[i]
        df_artist = df[df['country'] == artist]
        df_artist = df_artist.sort_values('year')
        fig.add_trace(go.Scatter(
            x=df_artist['year'].tolist(), y=df_artist['Porcentagem'].tolist(),
            mode='lines',
            name=artist,
            line=dict(width=0.5, color=colors[i]),
            stackgroup='one',
            groupnorm='percent'  # sets the normalization for the sum of the stackgroup
        ))

    fig.update_layout(
        showlegend=True,
        xaxis_type='category',
        title_text="Porcentagem de músicas ouvidas de cada país ao longo do tempo",
        yaxis=dict(
            type='linear',
            range=[1, 100],
            ticksuffix='%'))

    fig.write_image(filename)

def four(df, centers, filename):


    centers = np.array(centers)
    plt.subplots()
    sns.color_palette("tab10")
    sns.set_style()
    u_labels = df['prediction'].unique().tolist()
    labels = df['prediction'].tolist()
    color = {0: "blue",
              1: "green", 2: "orange", 3: "brown"}
    print("rotulos", df['prediction'].unique().tolist())
    features = df['features'].tolist()
    ages = [i[0] for i in features]
    age = [i[1] for i in features]

    df = pd.DataFrame({'Idade do usuário': ages, 'Idade': age, 'Grupo': labels})
    fig = sns.scatterplot(data=df, x="Idade do usuário", y="Idade", hue='Grupo')
    fig = fig.get_figure()
    plt.scatter(centers[:, 0], centers[:, 1], s=80, color='black')
    plt.xlabel("Idade do usuário")
    plt.ylabel("Idade da música")
    plt.title("Agrupamento de usuários com base nas suas idades e na idade das músicas")
    fig.savefig(filename)

if __name__ == '__main__':


    spark = SparkSession.builder.appName('abcd').getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    base_dir = "/home/claudio/Documentos/pdm/lastfm-dataset-1K/"

    schema = StructType([
        StructField("userid", StringType(), True),
        StructField("timestamp", TimestampType(), True),
        StructField("musicbrainz", StringType(), True),
    StructField("artist", StringType(), True),
    StructField("track", StringType(), True)])

    schema_2 = StructType([StructField("items", ArrayType(StringType()), True)])

    users_profile = spark.read.csv(base_dir + "userid-profile.tsv", sep=r'\t', inferSchema='true', header='true')
    users = spark.read.csv(base_dir + "userid-timestamp-artid-artname-traid-traname.tsv", sep=r'\t', schema=schema, header=False).sample(fraction=1., seed=1)
    users = users.join(users_profile, on='userid')
    users = users.withColumn('registered', col('registered').substr(9, 10))
    users = users.withColumn('registered', F.to_date(col('registered')))
    users = users.withColumn('duration_years', (col('timestamp') - col('registered')).cast('string').substr(11, 4).cast('int')/360)
    users.show(1)

    print(users.select('age').dropna().toPandas().describe())
    # Questão 1
    users_1 = users.select('artist', 'timestamp').dropna()

    print("10 artistas mais populares")
    top_10_artists = users_1.groupby('artist').count().sort('count', ascending=False)
    top_10_artists.show(10)
    top_10_artists = top_10_artists.select('artist')
    users_1_2 = users_1.filter(col('artist').isin([i[0] for i in spark.createDataFrame(top_10_artists.head(10)).toPandas().to_numpy()]))
    users_1_2 = users_1_2.dropna(subset=['artist', 'timestamp']).withColumn('year', F.year(F.to_date(col('timestamp'))))
    users_1_2_artist_year = users_1_2.groupby('artist', 'year').count().withColumn('artist_year_count', col('count')).select('artist', 'year', 'artist_year_count')
    users_1_2_year = users_1_2.groupby('year').count()
    users_1_2 = users_1_2_year.join(users_1_2_artist_year, on='year').withColumn('Porcentagem', col('artist_year_count') / col('count'))
    one(users_1.select('artist').groupby('artist').count().sort('count', ascending=False).toPandas(), users_1_2.toPandas(), 'artist', 'timestamp', "one_artist.png", "one_stacked_area.png")


    # features = ['artist', 'gender', 'country']
    # users_2 = users.select('artist', 'gender', 'country').filter(col('country') != "null").dropna(how='any', subset=features)
    # users_2 = users_2.withColumn('country', col('country') + " ")
    #
    # users_2 = users_2.withColumn('items', F.array(col('artist'), col('gender'), col('country')))
    # print("colunas", users_2.show(2))
    # fpGrowth = FPGrowth(itemsCol="items", minSupport=0.2, minConfidence=0.4)
    # model = fpGrowth.fit(users_2)
    #
    # # Display frequent itemsets.
    # model.freqItemsets.show()
    #
    # # Display generated association rules.
    # model.associationRules.show()
    #
    # # transform examines the input items against all the association rules and summarize the
    # # consequents as prediction
    # model.transform(users_2).show()

    # Questão 2
    users_2 = users.select('age', 'artist').dropna()
    top_artists = users_2.groupby('artist').count().orderBy('count', ascending=False)
    top_artists = spark.createDataFrame(top_artists.select('artist').head(10))
    users_2 = users_2.filter(col('artist').isin(list(top_artists.toPandas()['artist'].to_numpy())))

    users_2 = StringIndexer(inputCol='artist', outputCol='label').fit(users_2).transform(users_2)
    va = VectorAssembler(inputCols=['age'], outputCol='features')

    va_df = va.transform(users_2)
    va_df = va_df.select('features', 'label')

    (train, test) = va_df.randomSplit([0.8, 0.2], seed=10)

    dtc = DecisionTreeClassifier(featuresCol="features", labelCol="label")
    model = dtc.fit(train)

    pred = model.transform(test)

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                                  metricName="accuracy")
    accuracy = evaluator.evaluate(pred, params={evaluator.metricName: "f1"})

    prediction_list = pred.select('prediction').rdd.flatMap(lambda x: x).collect()
    label_list = pred.select('label').rdd.flatMap(lambda x: x).collect()

    print(classification_report(label_list, prediction_list,
                                target_names=list(top_artists.toPandas()['artist'].to_numpy())))
    dict_report = classification_report(label_list, prediction_list,
                                        target_names=list(top_artists.toPandas()['artist'].to_numpy()),
                                        output_dict=True)
    df_report = {'artist': [], 'precision': [], 'recall': [], 'f1-score': []}
    df_general = {'Métrica': [], 'Porcentagem': []}
    for key in dict_report:
        if key not in ['accuracy', 'macro avg', 'weighted avg']:
            df_report['artist'].append(key)
            df_report['precision'].append(dict_report[key]['precision'])
            df_report['recall'].append(dict_report[key]['recall'])
            df_report['f1-score'].append(dict_report[key]['f1-score'])
        else:
            df_general['Métrica'].append(key)
            if key == 'accuracy':
                df_general['Porcentagem'].append(dict_report[key])
            else:
                df_general['Porcentagem'].append(dict_report[key]['f1-score'])
    two_bar_one(pd.DataFrame(df_report))
    two_bar_two(pd.DataFrame(df_general))

    # Questão 3
    users_3_1 = users.select('country', 'timestamp').dropna().withColumn('year', F.year(F.to_date('timestamp'))).select('country', 'year')
    users_3_2 = users_3_1.groupby('country', 'year').count()
    users_3_1 = users_3_1.groupby('year').count()
    users_3_1 = users_3_1.withColumn('total', col('count')).select('year', 'total')
    users_3_1 = users_3_1.join(users_3_2, on='year').withColumn('Porcentagem', col('count')/col('total'))
    three_(users_3_1.sort('Porcentagem', ascending=False).toPandas(), "three.png")

    # Questão 4
    users_4 = users.select('age', 'duration_years').dropna(how='any')
    va = VectorAssembler(inputCols=['age', 'duration_years'], outputCol='features')

    va_df = va.transform(users_4)
    va_df = va_df.select('features')

    kmeans = KMeans(k=4).setSeed(1)
    model = kmeans.fit(va_df)

    # Make predictions
    predictions = model.transform(va_df)

    # Evaluate clustering by computing Silhouette score
    evaluator = ClusteringEvaluator()

    silhouette = evaluator.evaluate(predictions)
    predictions.show(10)
    print("Silhouette with squared euclidean distance = " + str(silhouette))

    centers = model.clusterCenters()

    print("Cluster Centers: ")
    for center in centers:
        print(center)

    four(predictions.sample(fraction=1., seed=1).toPandas(), centers, "four.png")
