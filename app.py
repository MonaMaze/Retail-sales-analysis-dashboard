#----------------------------# Load Your Dependencies#--------------------------#
import dash
from dash import  dcc    # Dash core Components
from dash import html   # HTML for Layout and Fonts
import plotly.express as px           # Plotly Graphs uses graph objects internally
import plotly.graph_objects as go     # Plotly Graph  more customized 
import pandas as pd                   # Pandas For Data Wrangling
import numpy as np
from dash import Input, Output  # Input, Output for  Call back functions

#--------------------------#Instanitiate Your App#--------------------------#

app = dash.Dash(__name__)  
server = app.server

#--------------------------# Pandas Section #------------------------------#

df_cust = pd.read_csv('Customers.csv') 
df_trans = pd.read_csv('Transactions.csv')
df_prod = pd.read_csv('prod_cat_info.csv')
df_cust_trans = pd.merge(left = df_cust , right = df_trans,
                          left_on = ('customer_Id'), right_on = ('cust_id'),
                          how = 'inner').drop('cust_id', axis =1)
df = pd.merge(left = df_cust_trans , right = df_prod,
                          left_on = ('prod_subcat_code', "prod_cat_code"), right_on = ('prod_sub_cat_code', "prod_cat_code"),
                          how = 'inner').drop('prod_sub_cat_code', axis =1)
for i in df.columns[df.isnull().any(axis=0)]: 
    df[i].fillna(df[i].mode()[0], inplace=True)
df = df.drop_duplicates()
#def myDateConv(tt):
#    sep = tt[2]
#    if sep == '-':
#        return pd.to_datetime(tt, format='%d-%m-%Y')
#    elif sep == '/':
#        return pd.to_datetime(tt, format='%m/%d/%Y')
#    else:
#        return tt
#df.tran_date = df.tran_date.apply(myDateConv)
#df.DOB = df.DOB.apply(myDateConv)
df['tran_date'] = pd.to_datetime(df['tran_date'], format="mixed")
df['trans_year'] =  pd.to_datetime(df['tran_date']).dt.year
df['trans_month'] =  pd.to_datetime(df['tran_date']).dt.month
df['DOB'] = pd.to_datetime(df['DOB'], format="mixed")
#df["DOB"].dt.strftime("%m/%d/%y")
#df['Age'] = pd.to_datetime(df['tran_date']).dt.year - pd.to_datetime(df['DOB']).dt.year
df['Age'] = (np.floor((pd.to_datetime(df['tran_date']) - pd.to_datetime(df['DOB'])).dt.days / 365.25)).astype(int)
df['age_cat'] = pd.cut(df['Age'], 3, labels=['young', 'adult', 'old'])
df_suc = df[df['Qty']>0]
df_ret = df[df['Qty']<0]
df_ret['Qty'] = abs(df_ret['Qty'])
df_ret['Rate'] = abs(df_ret['Rate'])
#--------------------------------------------------------------------------#
    
app.layout = html.Div([html.Div([html.A([html.H2('Retail Sales Dashboard'),html.Img(src='/assets/logo1.png')],  # A for hyper links
                                        href='http://projectnitrous.com/')],className="banner"),
                       # First raw
                       html.Div([
                           html.H4('Total revenue'),
                       ], className="three columns", style={'padding':10}),
                       html.Div([
                           html.H4('Sold quantity'),
                       ], className="three columns", style={'padding':10}),
                       html.Div([
                           html.H4('Returned quantity'),
                       ], className="three columns", style={'padding':10}),
                       # Total Revenue
                       html.Div([
                           dcc.RadioItems(
                               id='radio_revn',
                                options=[
                                    {'label': 'Annually', 'value': 'trans_year'},
                                    {'label': 'Monthly', 'value': 'trans_month'}
                                ],
                                value='trans_year',
                                labelStyle={'display': 'inline-block'}
                           ),
                           html.Div(
                               dcc.Graph(id='hist_revn'),
                           ),
                           #f6c3cb
                       ], className="three columns", style={'backgroundColor': '#efedfa', 'border-radius': 25, 'padding':10}),
                       # Sold Quantity
                       html.Div([
                           dcc.RadioItems(
                               id='radio_sale',
                                options=[
                                    {'label': 'Annually', 'value': 'trans_year'},
                                    {'label': 'Monthly', 'value': 'trans_month'}
                                ],
                                value='trans_year',
                                labelStyle={'display': 'inline-block'}
                           ),
                           html.Div(
                               dcc.Graph(id='hist_sale'),
                           ),
                       ], className="three columns", style={'backgroundColor': '#efedfa', 'border-radius': 25, 'padding':10}),
                       # Returned Quantity
                       html.Div([
                           dcc.RadioItems(
                               id='radio_ret',
                                options=[
                                    {'label': 'Annually', 'value': 'trans_year'},
                                    {'label': 'Monthly', 'value': 'trans_month'}
                                ],
                                value='trans_year',
                                labelStyle={'display': 'inline-block'}
                           ),
                           html.Div(
                               dcc.Graph(id='hist_ret'),
                           ),
                       ], className="three columns", style={'backgroundColor': '#efedfa', 'border-radius': 25, 'padding':10}),
                       
                       html.Div([html.Br(),],className="eleven columns"),
                       # Second raw
                       html.Div([
                           html.H4('Total revenue / city & gender'),
                       ], className="three columns", style={'padding':10}),
                       html.Div([
                           html.H4('Sold quantity / city & gender'),
                       ], className="three columns", style={'padding':10}),
                       html.Div([
                           html.H4('Display returned quantity '),
                       ], className="three columns", style={'padding':10}),
                       # Total Revenue/city & gender
                       html.Div([
                           dcc.Dropdown(
                                id='dropdown_revn',
                                options=[
                                    {'label': 'Gender', 'value': 'Gender'},
                                    {'label': 'City', 'value': 'city_code'},
                                ],

                                value=['Gender'],
                                multi=True,
                                searchable=True,
                                clearable=False
                           ),
                           html.Div(
                               dcc.Graph(id='sun_revn'),
                           ),
                       ], className="three columns", style={'backgroundColor': '#efedfa', 'border-radius': 25, 'padding':10}),
                       # Sold Quantity
                       html.Div([
                           dcc.Dropdown(
                                id='dropdown_sale',
                                options=[
                                    {'label': 'Gender', 'value': 'Gender'},
                                    {'label': 'City', 'value': 'city_code'},
                                ],

                                value=['Gender'],
                                multi=True,
                                searchable=True,
                                clearable=False
                           ),
                           html.Div(
                               dcc.Graph(id='sun_sale'),
                           ),
                       ], className="three columns", style={'backgroundColor': '#efedfa', 'border-radius': 25, 'padding':10}),
                       # Returned Quantity
                       html.Div([
                           dcc.RadioItems(
                               id='radio_cust',
                                options=[
                                    {'label': 'Count', 'value': 'count'},
                                    {'label': 'Percent', 'value': 'percent'}
                                ],
                                value='count',
                                labelStyle={'display': 'inline-block'}
                           ),
                           html.Div(
                               dcc.Graph(id='sun_cust'),
                           ),
                       ], className="three columns", style={'backgroundColor': '#efedfa', 'border-radius': 25, 'padding':10}),
                       
                       html.Div([html.Br(),],className="eleven columns"),
                       # Third raw
                       html.Div([
                           html.H4('Quantity/store type & products'),
                       ], className="three columns", style={'padding':10}),
                       html.Div([
                           html.H4('Quantity/store type & gender'),
                       ], className="three columns", style={'padding':10}),
                       html.Div([
                           html.H4('Quantity/store type & age'),
                       ], className="three columns", style={'padding':10}),
                       # Store type vs Products and Sub-Products
                       html.Div([
                           dcc.Dropdown(
                                id='dropdown_prod',
                                options=[
                                    {'label': 'Store type', 'value': 'Store_type'},
                                    {'label': 'Product', 'value': 'prod_cat'},
                                    {'label': 'Sub Product', 'value': 'prod_subcat'},
                                ],

                                value=['Store_type'],
                                multi=True,
                                searchable=True,
                                clearable=False
                           ),
                           html.Div(
                               dcc.Graph(id='sun_prod'),
                           ),
                       ], className="three columns", style={'backgroundColor': '#efedfa', 'border-radius': 25, 'padding':10}),
                       # Store type vs Gender
                       html.Div([
                           dcc.Dropdown(
                                id='dropdown_gen',
                                options=[
                                    {'label': 'Store type', 'value': 'Store_type'},
                                    {'label': 'Gender', 'value': 'Gender'},
                                ],

                                value=['Store_type'],
                                multi=True,
                                searchable=True,
                                clearable=False
                           ),
                           html.Div(
                               dcc.Graph(id='sun_gen'),
                           ),
                       ], className="three columns", style={'backgroundColor': '#efedfa', 'border-radius': 25, 'padding':10}),
                       # Store type vs Age
                       html.Div([
                           dcc.Dropdown(
                                id='dropdown_age',
                                options=[
                                    {'label': 'Store type', 'value': 'Store_type'},
                                    {'label': 'Age category', 'value': 'age_cat'},
                                ],

                                value=['Store_type'],
                                multi=True,
                                searchable=True,
                                clearable=False
                           ),
                           html.Div(
                               dcc.Graph(id='sun_age'),
                           ),
                       ], className="three columns", style={'backgroundColor': '#efedfa', 'border-radius': 25, 'padding':10}),
                       
                       html.Div([html.Br(),],className="eleven columns"),
                       # Fourth raw
                       html.Div([
                           html.H4('Customers count per city'),
                       ], className="five columns", style={'padding':10}),
                       html.Div([
                           html.H4('Correlations between different fields affecting clients'),
                       ], className="five columns", style={'padding':10}),
                       # City & No. of Customers
                       html.Div([
                               html.Div(['   '], style={'padding':18}),
                               dcc.Graph(id='hist_city'),
                       ], className="five columns", style={'backgroundColor': '#efedfa', 'border-radius': 25, 'padding':10}),
                       # Correlation Heatmap
                       html.Div([
                           dcc.Dropdown(
                                id='dropdown_cor',
                                options=[{'label':df.columns[i], 'value':df.columns[i]} for i in range(len(df.columns)) if (df[df.columns[i]].dtype== 'int64') or (df[df.columns[i]].dtype== 'float64')],

                                value=['Qty', 'Rate'],
                                multi=True,
                                searchable=True,
                                clearable=False
                           ),
                           html.Div(
                               dcc.Graph(id='heatmap_cor'),
                           ),
                       ], className="five columns", style={'backgroundColor': '#efedfa', 'border-radius': 25, 'padding':10}),
                       
                       html.Div([html.Br(),],className="eleven columns"),
                       # Fifth raw
                       # Sentences
                       html.Div([
                           html.H4('The most and least values'),
                       ], className="eleven columns", style={'padding':10}),
                       html.Div([
                           dcc.RadioItems(
                               id='radio_value',
                                options=[
                                    {'label': 'Highest value', 'value': 'most'},
                                    {'label': 'Lowest value', 'value': 'least'}
                                ],
                                value='most',
                                labelStyle={'display': 'inline-block'}
                           ),
                           html.H4(html.Div(id='sent1')),
                           html.H4(html.Div(id='sent2')),
                       ], className="eleven columns", style={'backgroundColor': '#efedfa', 'border-radius': 25, 'padding':10}),
               ])

@app.callback(
    Output('hist_revn', 'figure'),
    Input('radio_revn', 'value'),
    )
def revn_hist(value):
    fig = px.histogram(df, x=value, y="total_amt", color=value)
    fig.update_layout({'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
    return fig

@app.callback(
    Output('hist_sale', 'figure'),
    Input('radio_sale', 'value'),
    )
def sale_hist(value):
    fig = px.histogram(df_suc, x=value, y="Qty", color=value)
    fig.update_layout({'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
    return fig

@app.callback(
    Output('hist_ret', 'figure'),
    Input('radio_ret', 'value'),
    )
def sale_hist(value):
    fig = px.histogram(df_ret, x=value, y="Qty", color=value)
    fig.update_layout({'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
    return fig

@app.callback(
    Output('sun_revn', 'figure'),
    Input('dropdown_revn', 'value'),
    )
def revn_sun(value):
    fig = px.sunburst(data_frame=df, path=value, values='total_amt')
    fig.update_traces(textinfo='label+value')
    fig.update_layout({'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
    return fig

@app.callback(
    Output('sun_sale', 'figure'),
    Input('dropdown_sale', 'value'),
    )
def sale_sun(value):
    fig = px.sunburst(data_frame=df_suc, path=value, values='Qty')
    fig.update_traces(textinfo='label+value')
    fig.update_layout({'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
    return fig

@app.callback(
    Output('sun_cust', 'figure'),
    Input('radio_cust', 'value'),
    )
def cust_sun(value):
    fig = px.pie(data_frame=df, names='Gender')
    if value == 'count':
        fig.update_traces(textinfo="label+value")
    else:
        fig.update_traces(textinfo="label+percent")
    fig.update_layout({'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
    return fig

@app.callback(
    Output('sun_prod', 'figure'),
    Input('dropdown_prod', 'value'),
    )
def prod_sun(value):
    fig = px.sunburst(data_frame=df, path=value, values='Qty')
    fig.update_traces(textinfo='label+value')
    fig.update_layout({'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
    return fig

@app.callback(
    Output('sun_gen', 'figure'),
    Input('dropdown_gen', 'value'),
    )
def gen_sun(value):
    fig = px.sunburst(data_frame=df, path=value, values='Qty')
    fig.update_traces(textinfo='label+value')
    fig.update_layout({'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
    return fig

@app.callback(
    Output('sun_age', 'figure'),
    Input('dropdown_age', 'value'),
    )
def age_sun(value):
    fig = px.sunburst(data_frame=df, path=value, values='Qty')
    fig.update_traces(textinfo='label+value')
    fig.update_layout({'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
    return fig

@app.callback(
    Output('hist_city', 'figure'),
    Input('hist_city', 'id'),
    )
def revn_hist(value):
    fig = px.histogram(df, x='city_code', color='city_code')
    fig.update_layout({'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
    return fig

@app.callback(
    Output('heatmap_cor', 'figure'),
    Input('dropdown_cor', 'value'),
    )
def cor_heatmap(value):
    corr_matrix = df[value].corr()
    fig = px.imshow(corr_matrix, text_auto='.1f', aspect="auto", color_continuous_scale='RdBu_r')
    fig.update_layout({'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
    return fig

@app.callback(
    Output('sent1', 'children'),
    Input('radio_value', 'value'),
    )
def sum_value(value):
    #res = df['Age'].mode()[0]
    res = df['Age'].value_counts().index[0] if value == 'most' else df['Age'].value_counts().index[-1]
    return 'The customers aged between {} accounted for the {} sales'.format(res, value)

@app.callback(
    Output('sent2', 'children'),
    Input('radio_value', 'value'),
    )
def sum_value(value):
    pur = df_suc['total_amt'].max() if value == 'most' else df_suc['total_amt'].min()
    cust_id = df_suc[df_suc['total_amt']==pur].iloc[0]['customer_Id']
    qty = df_suc[df_suc['total_amt']==pur].iloc[0]['Qty']
    #gend = df[df['total_amt']==pur].iloc[0]['Gender']
    gen = 'male' if df_suc[df_suc['total_amt']==pur].iloc[0]['Gender'] == 'M' else 'female'
    age = int(df_suc[df_suc['total_amt']==pur].iloc[0]['Age'])
    city = df_suc[df_suc['total_amt']==pur].iloc[0]['city_code']
    nom = 'He' if df_suc[df_suc['total_amt']==pur].iloc[0]['Gender'] == 'M' else 'She'
    akk = 'His' if df_suc[df_suc['total_amt']==pur].iloc[0]['Gender'] == 'M' else 'Her'
    max_yr = df_suc[df_suc['customer_Id']==cust_id]['trans_year'].max()
    min_yr = df_suc[df_suc['customer_Id']==cust_id]['trans_year'].min()
    return 'Customer with ID {} spent the {} amount ${}, purchasing only {} items. {} is a {} years old, resident of city {}. {} purchases spanned across {} and {}'.format(cust_id, value, pur, qty, nom, age, city, akk, min_yr, max_yr)

if __name__ == '__main__':
    app.run_server()
