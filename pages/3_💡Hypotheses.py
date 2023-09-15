# ==========================================================
#                   Import libraries
# ==========================================================
import plotly.graph_objects as go
import plotly.express       as px
import matplotlib.pyplot    as plt
import streamlit            as st
import pandas               as pd
import geopandas

from urllib.error         import HTTPError
from datetime             import datetime
from PIL                  import Image


# ==========================================================
#                       Functions
# ==========================================================

@st.cache_data
def create_features(df_raw):
    df_raw['basement'] = df_raw.sqft_basement.apply(lambda x: 'No' if x==0 else 
                                                              'Yes')
    df_raw['standard'] = df_raw.apply(lambda row: 'low_standard' if (row.price<df_raw.price.quantile(0.75))&(row.condition!=5)&(row.grade<=10 ) else 
                                                  'high_standard', axis=1)
    df_raw['house_age'] = df_raw.yr_built.apply(lambda x: 'new_house' if x > 2014 else 
                                                          'old_house')
    df_raw['renovated'] = df_raw.yr_renovated.apply(lambda x: 'No' if x==0 else 
                                                              'Yes')
    dormitory_types = ['no_bedrooms', 'studio', 'apartament', 'house']
    df_raw['dormitory_type'] = [dormitory_types[min(x, 3)] for x in df_raw.bedrooms]
    df_raw['waterfront'] = df_raw.waterfront.apply(lambda x: 'No' if x==0 else 
                                                             'Yes')
    df_raw['price_per_sqft'] = df_raw.price / df_raw.sqft_living
    return df_raw

    
@st.cache_data
def categorize_features(df_raw):
    cols_to_categorize = ['id', 'waterfront', 'view', 'zipcode', 'season', 'basement', 'standard', 'house_age', 'renovated', 'dormitory_type']
    df_raw[cols_to_categorize] = df_raw[cols_to_categorize].apply(pd.Categorical)
    return df_raw


@st.cache_data
def insert_city_names(df_raw, _king_county):
    # create a GeoDataFrame with 'id' and the coordinates ('long', 'lat')
    geometry = geopandas.points_from_xy(df_raw.long, df_raw.lat)
    geo_df_raw = geopandas.GeoDataFrame(df_raw['id'], geometry=geometry)

    # checking if a point in 'geo_df_raw' is inside a polygon in 'king_county'
    pointInPolys = geopandas.tools.sjoin(geo_df_raw, 
                                         king_county, 
                                         op="within", 
                                         how='left')

    # merging 'pointInPolys' with the original 'df_raw'
    df_raw = pd.merge(df_raw, pointInPolys[['id', 'NAME']], how='left', on=['id', 'id'])
    df_raw.rename(columns={'NAME':'city'}, inplace=True)
    allowed_vals = ['Seattle', 'Northshore', 'Lake Washington', 'Federal Way', 'Highline', 'Tahoma', 'Bellevue','Riverview', 'Auburn', 'Mercer Island', 'Kent', 'Issaquah', 'Renton', 'Vashon Island', 'Snoqualmie Valley','Shoreline', 'Enumclaw', 'Tukwila', 'Fife', 'Skykomish']
    df_raw.loc[~df_raw["city"].isin(allowed_vals), "city"] = "undefined"
    df_raw['city'] = pd.Categorical(df_raw.city)
    
    return df_raw

    
@st.cache_data
def extract_features(df_raw):
    df_raw = create_features(df_raw)
    df_raw = categorize_features(df_raw)
    df_raw = insert_city_names(df_raw, king_county)

    # Reordering cols
    cols_order = ['id', 'date', 'season', 'year_sold', 'year_month', 'year_week', 'yr_built', 'house_age', 'yr_renovated', 'renovated',
                     'price', 'price_per_sqft', 'sqft_living', 'sqft_above', 'sqft_basement', 'basement', 'sqft_lot', 'sqft_living15', 'sqft_lot15',
                     'bedrooms', 'dormitory_type', 'bathrooms', 'floors', 'waterfront', 'view', 'condition', 'grade', 'standard',
                     'zipcode', 'lat', 'long', 'city']
    
    # Drop duplicates and sort by date
    df = (df_raw[cols_order]
          .drop_duplicates()
          .sort_values(by='date')
          .drop_duplicates(subset=['id'], keep='last'))
    
    return df


@st.cache_data
def feature_engineering(df_raw):
    # Some data transformation and feature engineering
    df_raw['date'] = df_raw['date'].str.replace('T000000', '')
    df_raw.date = pd.to_datetime(df_raw.date)
    df_raw['year_sold'] = df_raw.date.dt.year
    df_raw['year_month'] = df_raw.date.dt.strftime('%Y-%m')
    df_raw['year_week'] = df_raw.date.dt.strftime('%Y-%U')
    df_raw['date'] = df_raw['date'].dt.strftime('%Y-%m-%d')
    df_raw['season'] = ['Spring' if (x <= '2014-06-20') or (x >= '2015-03-20') else 
                       ('Summer' if (x >= '2014-06-21') and (x <= '2014-09-21') else 
                       ('Fall' if (x >= '2014-09-22') and (x <= '2014-12-20') else 
                        'Winter')) for x in df_raw.date]

    # correcting some errors
    # https://blue.kingcounty.com/Assessor/eRealProperty/default.aspx
    # to make some corrections each property with inconsistent data was searched on the site above to see what changes needed to be made
    replacements = {
        2569500210: {'bedrooms': 3},
        6306400140: {'bedrooms': 5, 'bathrooms': 4.5},
        3918400017: {'bedrooms': 3, 'bathrooms': 2.25},
        2954400190: {'bedrooms': 4, 'bathrooms': 4},
        2310060040: {'bedrooms': 4, 'bathrooms': 2.5},
        7849202190: {'bedrooms': 3, 'bathrooms': 1.5},
        9543000205: {'bedrooms': 2, 'bathrooms': 1},
        1222029077: {'bedrooms': 1, 'bathrooms': 1.5}
    }
    df_raw = df_raw.replace({'id': replacements})
    
    # what could not be changed was removed
    index_to_drop = [index for index in df_raw.index if df_raw.loc[index, 'bedrooms'] == 33 or 
                                                        df_raw.loc[index, 'bathrooms'] == 0 or 
                                                        df_raw.loc[index, 'bedrooms'] == 0]
    df_raw.drop(index_to_drop, inplace=True)
            
    return extract_features(df_raw)



def read_geopandas():
    url1 = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'
    url2 = 'https://opendata.arcgis.com/api/v3/datasets/e6c555c6ae7542b2bdec92485892b6e6_113/downloads/data?format=geojson&spatialRefId=4326&where=1%3D1'
    try:
        geofile = geopandas.read_file(url1)
        return geofile
    except HTTPError:
        try:
            geofile = geopandas.read_file(url2)
            return geofile
        except HTTPError:
            return None
    

def graph_price_feature(df, feature):
    df_aux = df[['price', feature]].groupby(feature).mean().reset_index()
    fig = px.bar(df_aux, y=feature, x='price', color='price',
                 color_continuous_scale='cividis',
                 title=f'Average prices  x  {df_aux.columns.tolist()[0]} ({df_aux.iloc[1,0]}/{df_aux.iloc[0,0]})')
           
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f'''##### The average price of buildings with {df_aux.columns.tolist()[0]}="{df_aux.iloc[1,0]}" is {(df_aux.price[1] / df_aux.price[0] - 1) * 100:.2f}% higher than that of buildings with {df_aux.columns.tolist()[0]}="{df_aux.iloc[0,0]}".''')
    st.dataframe(df_aux.round(2))
    return df_aux



import matplotlib.patches as mpatches
def plot_price_difference(df, feature):
    df_price = df[['price', feature]].groupby(feature).mean().reset_index()
    df_price['percentage_price_difference'] = df_price.price.pct_change()
    df_price['color'] = ['#fdea45' if x >= 0 else '#1f3e6e' for x in df_price['percentage_price_difference']]
    plt.figure(figsize=(20,5))
    ax = sns.barplot(data=df_price, x=feature, y='percentage_price_difference', palette=df_price['color'])
    ax.set_xlabel(f'# {feature}', fontsize=12)
    fig = plt.gcf()
    legend_labels = [mpatches.Patch(color='#fdea45', label='Increase'),
                     mpatches.Patch(color='#1f3e6e', label='Decrease')]
    plt.legend(handles=legend_labels)
    st.pyplot(fig, use_container_width=True)
    
    
    fig = px.bar(df_price, x=feature, y='price',
                 color='price', color_continuous_scale='cividis',
                 title=f'# {feature}  x  Avg Price',
                 text_auto='.2s',
                 labels={f"{feature}": f"# {feature}"}).update_yaxes(visible=False).update_coloraxes(showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    

def plot_map(df, feature, title):
    df = df.sort_values(feature)
    fig = px.scatter_mapbox(df,
                            title=f"{feature} distribution",
                            lat='lat',
                            lon='long',
                            size=feature,
                            color=feature,
                            color_continuous_scale='cividis',
                            size_max=20,
                            zoom=8.5)
    fig.update_layout(mapbox_style='open-street-map')

    fig.update_layout(height=600, margin={'r':0, 't':0, 'l':0, 'b':0})
    fig.update_layout(title_x=0.05, title_y=0.95, title_font_size=50, title_font_color='black', title_font_family='Arial')
    st.plotly_chart(fig, use_container_width=True)
    
    
    
    
def plot_price_difference_sqft(df, feature):
    mean_sqft = df.sqft_living.mean()

    conditions = ['sqft_living >  @mean_sqft', 'sqft_living <=  @mean_sqft']
    sqft_data = []
    for condition in conditions:
        sqft_data.append(df.query(condition))
        
    df_price_sqft = pd.DataFrame({
        'sqft_living': ['Above average', 'Below average'],
        'mean_price': [data.price.mean() for data in sqft_data]
    })
    st.dataframe(df_price_sqft.round(2))

    fig = px.bar(df_price_sqft,
                 y='sqft_living', x='mean_price', color='mean_price',
                 color_continuous_scale='cividis',
                 title='Average prices for properties with "below average" and "above average" areas')
    st.plotly_chart(fig, use_container_width=True)
           
    st.markdown(f'''#### Properties with sqft above average cost \${sqft_data[0].price.mean():.2f} on average, while those with sqft below average cost \${sqft_data[1].price.mean():.2f}''')
    st.markdown(f'''#### The average price of properties with a sqft above average is {(df_price_sqft.mean_price[0] / df_price_sqft.mean_price[1] - 1) * 100:.2f}% higher than that of properties with a sqft below average.''')
    
    plot_map(df, feature, "sqft_living distribution")
    
    
    
def plot_price_by_season(df):
    df_season_price = df[['season', 'price']].groupby('season').mean().sort_values('price').reset_index()
    df_season_price['price_difference_%'] = [np.nan if i == 0 else ((df_season_price.price[i] / df_season_price.price[0]) - 1) * 100 for i in range(len(df_season_price))]

    st.dataframe(df_season_price.reset_index(drop=True).round(2))
    fig = px.bar(df_season_price, x='price', y='season', color='price',
                   color_continuous_scale='cividis',
                   title='Average house prices by season')
    st.plotly_chart(fig, use_container_width=True)


    
    
def plot_price_by_month(df):
    df_price_month = df[['price', 'year_month']].groupby('year_month').mean().sort_values('year_month').reset_index().round(2)
    fig = px.line(df_price_month, y='price', x='year_month', 
                     title='Average monthly house prices',
                     markers=True)
    st.plotly_chart(fig, use_container_width=True)
    
    
    
    
def variation_city_prices(df):
    df_city_price = df.groupby('city')['price'].agg(['mean', 'std']).reset_index()
    df_city_price.rename(columns={'mean': 'mean_price', 'std': 'std_price'}, inplace=True)
    df_city_price['var_%'] = df_city_price.std_price / df_city_price.mean_price * 100
    df_city_price.set_index('city', inplace=True)
    df_city_price = df_city_price.round(2).sort_values('var_%', ascending=False).T
    st.dataframe(df_city_price)
    
    
    
def plot_price_renovated(df):
    df_price_renovated = df.groupby(['yr_built', 'renovated']).mean().reset_index()

    fig = go.Figure()
    for renovated in ['No', 'Yes']:
        fig.add_trace(go.Scatter(
            x=df_price_renovated.query(f'renovated == "{renovated}"').yr_built,
            y=df_price_renovated.query(f'renovated == "{renovated}"').price.tolist(),
            mode='lines',
            name=f'renovated {renovated}',
            marker_color='#1f3e6e' if renovated == 'No' else '#beb200'
        ))

    fig.update_layout(barmode='group', xaxis_tickangle=-45, title='Average sale price over the years for "renovated" and "not renovated" properties')
    st.plotly_chart(fig, use_container_width=True)


    
    
    
def plot_price_per_sqft_dormitory_type(df):
    df_price_per_sqft_dormitory_type = df.groupby('dormitory_type').mean().sort_values('price_per_sqft').reset_index()
    st.dataframe(df_price_per_sqft_dormitory_type.round(2))
    fig = px.bar(df_price_per_sqft_dormitory_type, 
                 x='price_per_sqft', y='dormitory_type', color='price_per_sqft',
                 color_continuous_scale=['#1f3e6e','#beb200'],
                 title='Average price/sqft by "dormitory_type"')
    st.plotly_chart(fig, use_container_width=True)


# ----------------------------- Start of the logical code structure -----------------------------

st.set_page_config(page_title='Hypotheses',
                   page_icon='💡',
                   layout='wide',
                   initial_sidebar_state="expanded"
                  )

# -------------------
# Import Datasets
# -------------------
kc_house_data = pd.read_csv("dataset/kc_house_data.csv")
king_county = geopandas.read_file('dataset/King_County_Shape_File/School_Districts_in_King_County___schdst_area.shp')
geofile = read_geopandas()


# -------------------
# Feature engineering
# -------------------
data = feature_engineering(kc_house_data)




# ==========================================================
#                       Sidebar
# ==========================================================
image = Image.open('images/logo.png')
st.sidebar.image(image, width=250)

st.sidebar.markdown('''
- [Hypothesis 1](#hypothesis-1)
- [Hypothesis 2](#hypothesis-2)
- [Hypothesis 3](#hypothesis-3)
- [Hypothesis 4](#hypothesis-4)
- [Hypothesis 5](#hypothesis-5)
- [Hypothesis 6](#hypothesis-6)
- [Hypothesis 7](#hypothesis-7)
- [Hypothesis 8](#hypothesis-8)
- [Hypothesis 9](#hypothesis-9)
- [Hypothesis 10](#hypothesis-10)
''')

st.sidebar.markdown('''___''')
st.sidebar.markdown('##### Powered by Comunidade DS')
st.sidebar.markdown('##### Data Analyst: Daniel Gomes')
# ==========================================================
#                Layout in Streamlit
# ==========================================================

image = Image.open('images/real-estate-agent-showing-house-plan-to-buyer.jpg')
st.image(image, 
         caption='https://www.freepik.com/free-photo/real-estate-agent-showing-house-plan-buyer_3955605.htm?log-in=google. Image by @yanalya on Freepik'
        )

st.title(''' Answers to the hypotheses assumed
## During the analysis process, the business team came up with some hypotheses, which are presented below.

''')

st.header(':blue[Hypothesis 1]')
st.markdown('''### - :blue[Houses with basements are 20% more expensive than houses without basements.]
> #### :blue[Hypothesis TRUE. Properties with basements have an average price 27% higher than properties without basements.]
''')
graph_price_feature(data, 'basement')


st.header(':blue[Hypothesis 2]')
st.markdown('''### :blue[Property prices increase with each increase in the number of bathrooms.]
> #### :blue[Hypothesis FALSE. According to the graph below, the percentage change is not positive for each increase in the number of bathrooms.]
''')
plot_price_difference(data, 'bathrooms')


st.header(':blue[Hypothesis 3]')
st.markdown('''### :blue[Property prices increase with each increase in the number of bedrooms.]
> #### :blue[Hypothesis FALSE. According to the graph below, the percentage change is not positive for each increase in the number of bedrooms.]
''')
plot_price_difference(data, 'bedrooms')


st.header(':blue[Hypothesis 4]')
st.markdown('''### :blue[The average price of properties above the average price per sqft is 50% higher than properties below the average price per sqft.]
> #### :blue[Hypothesis TRUE. Properties with an area above the average have prices around 88% higher than properties with an area below the average price.]
''')
plot_price_difference_sqft(data[['sqft_living', 'price', 'id', 'lat', 'long']], 'sqft_living')


st.header(':blue[Hypothesis 5]')
st.markdown('''### :blue[On average, houses with a water view are 25% more expensive than others.]
> #### :blue[Hypothesis TRUE. Properties with a water view cost on average 211% more than properties without a water view.]
''')
plot_map(data[['waterfront', 'price', 'id', 'lat', 'long']].loc[data.waterfront == 'Yes'], 'price', "Waterfront houses")
graph_price_feature(data, 'waterfront')


st.header(':blue[Hypothesis 6]')
st.markdown('''### :blue[In winter, average prices are at least 10% lower than in other seasons.]
> #### :blue[Hypothesis FALSE. Prices don't vary by even 7% between winter and the other seasons.]
''')
plot_price_by_season(data[['season', 'price']])
plot_price_by_month(data[['price', 'year_month']])


st.header(':blue[Hypothesis 7]')
st.markdown('''### :blue[The variation in prices within the same city is less than 50%.]
> #### :blue[Hypothesis FALSE. Although in most cities the variation is less than 50%, in some there are variations of up to 70%.]
''')
variation_city_prices(data[['price', 'city']])


st.header(':blue[Hypothesis 8]')
st.markdown('''### :blue[Renovated properties are at least 40% more expensive than those of the same year without renovation.]
> #### :blue[Hypothesis FALSE. As can be seen in the graph, there are construction years in which renovated properties are cheaper than non-refurbished properties (1903 and 1991, for example). ]
> #### :blue[Furthermore, even when renovated properties are more expensive, the percentage varies and in some cases is less than 40%, as in 1900, when the difference is 5.13%.]
''')
plot_price_renovated(data[['price', 'yr_built', 'renovated']])
graph_price_feature(data, 'renovated')


st.header(':blue[Hypothesis 9]')
st.markdown('''### :blue[The price/sqft of STUDIO properties are 30% higher than HOUSE properties.]
> #### :blue[Hypothesis TRUE. On average, STUDIO properties have price/sqft values 54% higher than HOUSE properties.]
''')
plot_price_per_sqft_dormitory_type(data[['price_per_sqft', 'dormitory_type']])


st.header(':blue[Hypothesis 10]')
st.markdown('''### :blue[The STUDIO price/sqft is 15% higher than the APARTMENT.]
> #### :blue[Hypothesis TRUE. On average, STUDIO properties have 17% higher price/sqft than APARTMENT properties.]
''')