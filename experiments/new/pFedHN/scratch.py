
import numpy as np

import matplotlib.pyplot as plt


# from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_blobs
# import plotly
# import plotly.graph_objs as go
# from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
# from experiments.new.node import BaseNodes
# def visualize_3d(X, y, algorithm="tsne", title="Data in 3D"):
#     from sklearn.manifold import TSNE
#     from sklearn.decomposition import PCA
#
#     if algorithm == "tsne":
#         reducer = TSNE(n_components=3, random_state=47, n_iter=400, angle=0.6)
#     elif algorithm == "pca":
#         reducer = PCA(n_components=3, random_state=47)
#     else:
#         raise ValueError("Unsupported dimensionality reduction algorithm given.")
#
#     if X.shape[1] > 3:
#         X = reducer.fit_transform(X)
#     else:
#         if type(X) == pd.DataFrame:
#             X = X.values
#
#     marker_shapes = ["circle", "diamond", "circle-open", "square", "diamond-open", "cross", "square-open", ]
#     traces = []
#     for hue in np.unique(y):
#         X1 = X[y == hue]
#
#         trace = go.Scatter3d(
#             x=X1[:, 0],
#             y=X1[:, 1],
#             z=X1[:, 2],
#             mode='markers',
#             name=str(hue),
#             marker=dict(
#                 size=12,
#                 symbol=marker_shapes.pop(),
#                 line=dict(
#                     width=int(np.random.randint(3, 10) / 10)
#                 ),
#                 opacity=int(np.random.randint(6, 10) / 10)
#             )
#         )
#         traces.append(trace)
#
#     layout = go.Layout(
#         title=title,
#         scene=dict(
#             xaxis=dict(
#                 title='Dim 1'),
#             yaxis=dict(
#                 title='Dim 2'),
#             zaxis=dict(
#                 title='Dim 3'), ),
#         margin=dict(
#             l=0,
#             r=0,
#             b=0,
#             t=0
#         )
#     )
#     fig = go.Figure(data=traces, layout=layout)
#     iplot(fig)
#
# # Dataset
# from sklearn.datasets import make_classification
# def read_dataset(path, data_types, data_name):
#     if data_name == 'adult':
#         data = pd.read_csv(
#             path,
#             names=data_types,
#             index_col=None,
#             dtype=data_types,
#             comment='|',
#             skipinitialspace=True,
#             na_values={
#                 'capital_gain':99999,
#                 'workclass':'?',
#                 'native_country':'?',
#                 'occupation':'?',
#             },
#         )
#     elif data_name == 'compas':
#         url = 'https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv'
#         data = pd.read_csv(url)
#     return data
#
# def clean_and_encode_dataset(data, data_name):
#     if data_name == 'adult':
#         data['income_class'] = data.income_class.str.rstrip('.').astype('category')
#
#         data = data.drop('final_weight', axis=1)
#
#         data = data.drop_duplicates()
#
#         data = data.dropna(how='any', axis=0)
#
#         data.capital_gain = data.capital_gain.astype(int)
#
#         data.replace(['Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent', 'Never-married',
#                       'Separated', 'Widowed'],
#                      ['not married', 'married', 'married', 'married', 'not married', 'not married', 'not married'],
#                      inplace=True)
#         data.replace(['Federal-gov', 'Local-gov', 'State-gov'], ['government', 'government', 'government'],
#                      inplace=True)
#         data.replace(['Never-worked', 'Private', 'Self-emp-inc', 'Self-emp-not-inc', 'Without-pay'],
#                      ['private', 'private', 'private', 'private', 'private'], inplace=True)
#         encoders = {}
#
#         for col in ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country', 'income_class']:
#             encoders[col] = LabelEncoder().fit(data[col])
#             data.loc[:, col] = encoders[col].transform(data[col])
#
#     elif data_name == 'compas':
#         data = data.loc[data['days_b_screening_arrest'] <= 30]
#         data = data.loc[data['days_b_screening_arrest'] >= -30]
#         data = data.loc[data['is_recid'] != -1]
#         data = data.loc[data['c_charge_degree'] != "O"]
#         data = data.loc[data['score_text'] != 'N/A']
#         #data['race'].loc[data['race'] != "Caucasian"] = 'Other'
#         data['is_med_or_high_risk'] = (data['decile_score'] >= 5).astype(int)
#         data['length_of_stay'] = (
#                 pd.to_datetime(data['c_jail_out']) - pd.to_datetime(data['c_jail_in']))
#
#         #cols = ['age', 'c_charge_degree', 'sex', 'age_cat', 'score_text', 'race', 'priors_count', 'length_of_stay', 'days_b_screening_arrest', 'decile_score', 'two_year_recid']
#         cols = ['race', 'age_cat','sex','two_year_recid']
#
#         data = data[cols]
#
#
#         encoders = {}
#
#         for col in ['sex','race', 'age_cat']:
#             encoders[col] = LabelEncoder().fit(data[col])
#             data.loc[:, col] = encoders[col].transform(data[col])
#
#     return data
#
# CURRENT_DIR = os.path.abspath(os.path.dirname(__name__))
# TRAIN_DATA_FILE = os.path.join(CURRENT_DIR, 'adult.data')
# TEST_DATA_FILE = os.path.join(CURRENT_DIR, 'adult.test')
#
# data_types = OrderedDict([
#     ("age", "int"),
#     ("workclass", "category"),
#     ("final_weight", "int"),
#     ("education", "category"),
#     ("education_num", "int"),
#     ("marital_status","category"),
#     ("occupation", "category"),
#     ("relationship", "category"),
#     ("race" ,"category"),
#     ("sex", "category"),
#     ("capital_gain", "float"),
#     ("capital_loss", "int"),
#     ("hours_per_week", "int"),
#     ("native_country", "category"),
#     ("income_class", "category"),
# ])
#
# train_data = clean_and_encode_dataset(read_dataset(TRAIN_DATA_FILE, data_types, 'adult'), 'adult')  #(32561, 14)
#
# cols = train_data.columns
# features, labels = cols[:-1], cols[-1]


#
# X,y = make_classification(n_samples=10000, n_features=3, n_informative=3,
#                     n_redundant=0, n_repeated=0, n_classes=3, n_clusters_per_class=2,
#                           class_sep=1.5,
#                    flip_y=0,weights=[0.5,0.5,0.5])
# X = pd.DataFrame(X)
# y = pd.Series(y)
# visualize_3d(X,y)

#
# centers = [(-.5, -.5, -.5), (.5, .5, .5), (-.5, -.5, 5)]
# data = make_blobs(n_samples=1000, n_features=3, centers=centers, )
#
# # Data for a three-dimensional line
# zline = np.linspace(0, 15, 1000)
# xline = np.sin(zline)
# yline = np.cos(zline)
# ax.plot3D(xline, yline, zline, 'gray')

# Data for three-dimensional scattered points
zdata = 3* np.random.random(size=1000)
zdata = zdata.astype(int)
y_data = 4 * np.random.random(size=1000)
xdata, labels = make_blobs(n_samples=1000, n_features= 3, cluster_std=.1,center_box=(-.1,.1), centers=[(0, 0, 0), (1,1,1)])
colors = []


# '#6284e3' in
# '#000000' out
white_accepted = [0,0,0,0]
black_accepted = [0,0,0,0]
other_accepted = [0,0,0,0]

male_accepted = 0
female_accepted = 0

for i, x in enumerate(abs(xdata[:,0])):
    print(i, x)
    if x <= .5 and zdata[i] <= .25:
        colors.append('#6284e3')
        if y_data[i] <= 1:
            white_accepted[0] = white_accepted[0] + 1
        elif y_data[i] <=2 and y_data[i] > 1:
            white_accepted[1] = white_accepted[1] + 1
        elif y_data[i] <=3 and y_data[i] > 2:
            white_accepted[2] = white_accepted[2] + 1
        elif y_data[i] > 3:
            white_accepted[3] = white_accepted[3] + 1

        male_accepted +=1
    elif x <= .5 and y_data[i] <= 3 and zdata[i] > .75 and zdata[i] < 1.25:
        colors.append('#e36262')
    elif x <= .5 and y_data[i] > 3 and zdata[i] > .75 and zdata[i] < 1.25:
        colors.append('#6284e3')
        if y_data[i] <= 1:
            black_accepted[0] = black_accepted[0] + 1
        elif y_data[i] <= 2 and y_data[i] > 1:
            black_accepted[1] = black_accepted[1] + 1
        elif y_data[i] <= 3 and y_data[i] > 2:
            black_accepted[2] = black_accepted[2] + 1
        elif y_data[i] > 3:
            black_accepted[3] = black_accepted[3] + 1
        male_accepted += 1
    elif x <= .5 and y_data[i] <= 1 and zdata[i] > 1.25:
        colors.append('#e36262')
    elif x <= .5 and y_data[i] > 1 and zdata[i] > 1.25:
        colors.append('#6284e3')
        if y_data[i] <= 1:
            other_accepted[0] = other_accepted[0] + 1
        elif y_data[i] <= 2 and y_data[i] > 1:
            other_accepted[1] = other_accepted[1] + 1
        elif y_data[i] <= 3 and y_data[i] > 2:
            other_accepted[2] = other_accepted[2] + 1
        elif y_data[i] > 3:
            other_accepted[3] = other_accepted[3] + 1
        male_accepted += 1

    elif x > .5 and y_data[i] <= 1 and zdata[i] <= .25:
        colors.append('#e36262')
    elif x > .5 and y_data[i] > 1 and zdata[i] <= .25:
        colors.append('#6284e3')
        if y_data[i] <= 1:
            white_accepted[0] = white_accepted[0] + 1
        elif y_data[i] <= 2 and y_data[i] > 1:
            white_accepted[1] = white_accepted[1] + 1
        elif y_data[i] <= 3 and y_data[i] > 2:
            white_accepted[2] = white_accepted[2] + 1
        elif y_data[i] > 3:
            white_accepted[3] = white_accepted[3] + 1
        female_accepted +=1
    elif x > .5 and zdata[i] > .75 and zdata[i] < 1.25:
        colors.append('#e36262')
    elif x > .5 and y_data[i] <= 2 and zdata[i] > 1.25:
        colors.append('#e36262')
    elif x > .5 and y_data[i] > 2 and zdata[i] > 1.25:
        colors.append('#6284e3')
        if y_data[i] <= 1:
            other_accepted[0] = other_accepted[0] + 1
        elif y_data[i] <= 2 and y_data[i] > 1:
            other_accepted[1] = other_accepted[1] + 1
        elif y_data[i] <= 3 and y_data[i] > 2:
            other_accepted[2] = other_accepted[2] + 1
        elif y_data[i] > 3:
            other_accepted[3] = other_accepted[3] + 1
        female_accepted += 1

# my_path = r"C:\Users\ancarey\PycharmProjects\adaptive-hypernets\experiments\new\pFedHN\intersectionality.png"
# my_file = "intersectionality.png"
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
fig = plt.figure()
ax = plt.axes(projection='3d')
plt.rcParams['figure.dpi'] = 600
ax.scatter3D(abs(xdata[:,0]), y_data, zdata, c= colors)
ax.set_xticks([0, 1])
ax.set_zticks([0,1,2])
ax.set_xlabel('Gender', fontsize=10)
ax.set_ylabel('ACT Score', fontsize=10)
ax.set_zlabel('Race', fontsize=10)
ax.set_title('Acceptance to CS Department')
#plt.xticks(np.arange(0,1, step=1), ['Male', 'Female'])
plt.show()


objects = ['Male', 'Female']
y_pos = np.arange(len(objects))
performance = [male_accepted, female_accepted]
plt.bar(y_pos, performance, align='center', alpha=.8, color='#5f9dc9')
plt.xticks(y_pos, objects, fontsize=15)
plt.xlabel('Gender', fontsize=15)
plt.ylabel('ACT Score', fontsize=15)
plt.title('Acceptance to CS Department by Gender and ACT Score', fontsize=15)
plt.show()

# race
rects1 = plt.bar(np.arange(4), white_accepted, .3,
alpha=0.8,
color='#82c95f',
label='White')

rects2 = plt.bar(np.arange(4) + .3, black_accepted, .3,
alpha=0.8,
color='#5f9dc9',
label='Black')

rects3 = plt.bar(np.arange(4) + .6, other_accepted, .3,
alpha=0.8,
color='#8c60b5',
label='Other')

plt.xticks( np.arange(4) + .3, ['Poor','Okay','Average','Excelent'], fontsize=15)
plt.legend()
plt.xlabel('Race', fontsize=15)
plt.ylabel('ACT Score', fontsize=15)
plt.title('Acceptance to CS Department by Race and ACT Score', fontsize=15)
plt.show()