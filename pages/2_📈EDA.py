# ==========================================================
#                   Import libraries
# ==========================================================
import plotly.graph_objects as go
import plotly.express       as px
import matplotlib.pyplot    as plt
import seaborn              as sns
import streamlit            as st
import pandas               as pd
import numpy                as np
import folium
import geopandas

from urllib.error         import HTTPError
from streamlit_folium     import folium_static
from folium.plugins       import HeatMap
from datetime             import datetime
from plotly.subplots      import make_subplots
from PIL                  import Image

# ==========================================================
#                       Functions
# ==========================================================


@st.cache_data
def create_features(df_raw):
    df_raw['basement'] = df_raw.sqft_basement.apply(lambda x: 'No' if x==0 else 'Yes')
    df_raw['standard'] = df_raw.apply(lambda row: 'low_standard' if (row.price<df_raw.price.quantile(0.75))&(row.condition!=5)&(row.grade<=10 ) else                                                             'high_standard', 
                                      axis=1)
    df_raw['house_age'] = df_raw.yr_built.apply(lambda x: 'new_house' if x > 2014 else 'old_house')
    df_raw['renovated'] = df_raw.yr_renovated.apply(lambda x: 'No' if x==0 else 'Yes')
    dormitory_types = ['no_bedrooms', 'studio', 'apartament', 'house']
    df_raw['dormitory_type'] = [dormitory_types[min(x, 3)] for x in df_raw.bedrooms]
    df_raw['waterfront'] = df_raw.waterfront.apply(lambda x: 'No' if x==0 else 'Yes')
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
    
    
    
@st.cache_data
def avg_values(data):
# Average metrics
    df1 = data[['id', 'city']].groupby( 'city' ).count().reset_index()
    df2 = data[['price', 'city']].groupby( 'city').mean().reset_index()
    df3 = data[['sqft_living', 'city']].groupby( 'city').mean().reset_index()
    df4 = data[['price_per_sqft', 'city']].groupby( 'city').mean().reset_index()

    # merge
    m1 = pd.merge( df1, df2, on='city', how='inner' )
    m2 = pd.merge( m1, df3, on='city', how='inner' )
    df = pd.merge( m2, df4, on='city', how='inner' )

    df.columns = ['CITY', '# HOUSES', 'AVG PRICE', 'AVG SQRT LIVING', 'AVG PRICE/SQFT']
    
    return df


ZOOM_LEVEL = 8.5
def portfolio_density(data, geofile):
    st.markdown("<h3 style='text-align: center; color: #129fa5;'>Portfolio Density</h3>",
                unsafe_allow_html=True)
    density_map = folium.Map(location = [data.lat.mean(), data.long.mean()],
                             zoom_start = ZOOM_LEVEL,
                             control_scale = True)
    HeatMap(data[['lat', 'long', 'price']], 
            min_opacity=0.1,
            blur = 18).add_to(folium.FeatureGroup(name='Heat Map').add_to(density_map))
    folium_static(density_map, width=600)
    return None



def price_density_map(data, geofile):
    st.markdown("<h3 style='text-align: center; color: #129fa5;'>Price Density</h3>",
                  unsafe_allow_html=True)
    df = data[["price", "zipcode"]].groupby("zipcode").mean().reset_index()
    df.columns = ["ZIP", "PRICE"]

    geofile = geofile[geofile['ZIP'].isin(df['ZIP'].tolist())]
    region_price_map = folium.Map(location=[data.lat.mean(), data.long.mean()], 
                                  zoom_start = ZOOM_LEVEL)
    folium.Choropleth(
        data=df,
        geo_data=geofile,
        columns=['ZIP', 'PRICE'],
        key_on='feature.properties.ZIP',
        fill_color='Spectral_r',
        bins=12,
        fill_opacity = 0.6,
        line_opacity = 0.2,
        legend_name = 'AVG PRICE'
    ).add_to(region_price_map)
    folium_static(region_price_map, width=600)
    return None



def create_scatter_plot(dataframe, x_col, y_col, color, name):
    return go.Scatter(x=dataframe[x_col],
                      y=dataframe[y_col],
                      name=name,
                      line=dict(color=color))



def average_price_per_year(data, f_year_built):
    # ---------------Average price per year---------------
    st.header('Average price per year of built and year of renovation')
    # data selection
    df = data.loc[data.yr_built < f_year_built]
    
    fig = make_subplots(rows=2, cols=1,
                        shared_xaxes=True,
                       )
    
    price_by_yr_built = df[['yr_built', 'price']].groupby('yr_built').mean().reset_index()
    fig.append_trace(create_scatter_plot(price_by_yr_built, 
                                         'yr_built', 
                                         'price', 
                                         'firebrick', 
                                         'Average price per year of built'),
                     row=1, col=1)
    
    yr_renovated_after_1930 = df.query('yr_renovated >= 1930')
    price_by_yr_renovated = yr_renovated_after_1930[['price', 'yr_renovated']].groupby('yr_renovated').mean().reset_index()
    fig.append_trace(create_scatter_plot(price_by_yr_renovated, 
                                         'yr_renovated', 
                                         'price', 
                                         'royalblue', 
                                         'Average price per year of renovation'), 
                     row=2, col=1)
    # plot
    st.plotly_chart(fig, use_container_width=True)

    
    
    
def average_price_per_day(data, f_date):
    # ---------------Average price per day---------------
    st.header('Average price per day')
    # data filtering
    data.date = pd.to_datetime(data.date)
    df = data.loc[data.date < f_date]
    df = df[['date', 'price']].groupby('date').mean().reset_index()
    fig = px.line(df, x='date', y='price')
    st.plotly_chart(fig, use_container_width=True)


    

def create_histogram(data, column, title):
    st.markdown(f'### {title}')
    fig = px.histogram(data, x=column)
    st.plotly_chart(fig, use_container_width=True)

def attributes_distribution(data, f_bedrooms, f_bathrooms, f_floors, f_waterview):
    with st.container():
        col1, col2 = st.columns(2, gap='large')
        with col1:
            create_histogram(data[data.floors < f_floors], 'floors', 'Houses per floor')
        with col2:
            create_histogram(data[data.bathrooms < f_bathrooms], 'bathrooms', 'Houses per bathrooms')
    
    create_histogram(data[data.bedrooms < f_bedrooms], 'bedrooms', 'Houses per bedrooms')

    return None


@st.cache_data
def price_distribution(df):
    houses = df.sort_values('price')
    houses['over_1_million'] = ['No' if x < 1000000 else 'Yes' for x in houses.price]
    
    fig = px.scatter_mapbox(houses,
                            title="<b>Geographical<br>Price Distribution</b>",
                            lat='lat',
                            lon='long',
                            size='price',
                            color_discrete_sequence=['#1f3e6e','#fdea45'],
                            color='over_1_million',
                            size_max=15,
                            zoom=ZOOM_LEVEL)

    fig.update_layout(mapbox_style='open-street-map')
    fig.update_layout(height=600, margin={'r':0, 't':0, 'l':0, 'b':0})
    fig.update_layout(title_x=0.05, title_y=0.9, title_font_size=50, title_font_color='black', title_font_family='Arial')
    st.plotly_chart(fig, use_container_width=True)
    
    plt.figure(figsize=(25,6))
    ax = sns.histplot(data=houses, x='price', hue='over_1_million', palette=["#1f3e6e", "#fdea45"])
    ax.set_title('Price histogram', fontsize=25)
    ax.set_xlabel('Price($)', fontsize=18)
    ax.set_ylabel('Count', fontsize=18)
    fig = plt.gcf()
    st.pyplot(fig, use_container_width=True)
    
    st.markdown("#### It's possible to notice a high concentration of properties with prices below $1,000,000 which is equivalent to"+ 
      f" {len(df.query('price <= 1000000')) / df.shape[0] * 100:.0f}% of the dataset.")
    
    price_by_standard = df[['price', 'standard']].groupby('standard')['price'].sum().reset_index()
    st.plotly_chart(px.bar(price_by_standard,
                           y='standard', x='price', color='price',
                           color_continuous_scale='cividis',
                          ), use_container_width=True)
    
    len_high_standard = len(df.query('standard == "high_standard"'))
    st.markdown(f"#### Based on the graph above, it is possible to observe that high-standard* properties are equivalent to {len_high_standard/df.shape[0]*100:.2f}% of the database, but the sum of the prices of this type of properties is just {(1-(price_by_standard['price'][0]/price_by_standard['price'][1]))*100:.2f}% lower than __low-standard__ properties, in other words, the price volume is pretty much the same.")
    
    st.markdown('''
    ##### \* High standard properties:
        - prices = 25% biggest prices
        - condition = 5
        - grade > 10
        ''')
    
    return None




@st.cache_data
def correlation_map(data):
    plt.figure(figsize=(20, 6))
    correlation_matrix = data.select_dtypes(include=['float64', 'int64']).corr()
    mask = np.triu(np.ones_like(correlation_matrix))
    sns.heatmap(correlation_matrix, 
                cmap='cividis', 
                # https://colorbrewer2.org/#type=diverging&scheme=RdBu&n=11
                annot=True, mask = mask, vmin=-0.9, vmax=0.9
               )
    plt.title('Correlation Map', fontsize=25, fontweight = 'bold')
    fig = plt.gcf()
    st.pyplot(fig, use_container_width=True)
    st.markdown(
        '''
        #### The variables most correlated with price are:
            * price_per_sqft
            * sqft_living
            * sqft_above
            * sqft_living15
            * bathrooms
            * grade
         ''')
    st.markdown('#### Based on this, it is possible to notice that the prices are more related to physical characteristics, such as the size of the interior living space and the number of bathrooms.')
    return None




@st.cache_data
def info_dormitory(data):
    dormitory_info = data.groupby(by='dormitory_type')['price_per_sqft'].describe().sort_values(by='count', ascending=False).reset_index()
    st.dataframe(dormitory_info)

    st.plotly_chart(px.bar(dormitory_info.sort_values(by='count'),
                           y='dormitory_type', x='count', color='count',
                           color_continuous_scale='cividis',
                           title='Number of properties by dormitory type',
                           labels={"mean": "price/sqft mean",
                                   'dormitory_type':'Dormitory Type'}), 
                    use_container_width=True)

    st.plotly_chart(px.bar(dormitory_info.sort_values(by='count'),
                           y='dormitory_type', x='mean', color='mean',
                           color_continuous_scale='cividis',
                           title='Average price/sqft by dormitory type',
                           labels={"mean": "price/sqft mean",
                                   'dormitory_type':'Dormitory Type'}), 
                    use_container_width=True)
    
    st.markdown(f'#### The database is mainly composed of houses ({dormitory_info["count"][0] / data.shape[0] * 100:.2f}% of total).')
    st.markdown('#### Properties with fewer bedrooms have, on average, a higher price/sqft despite having a lower market value.')
    
    plt.figure(figsize=(20,6))
    ax = sns.boxplot(data=data, x='dormitory_type', y='price', hue='standard', 
                showfliers=False, palette=["#1f3e6e", "#fdea45"], order=["house", "apartament", "studio"])
    ax.set_title('Apartment prices by dormitory type and level', fontsize=25)
    ax.set_ylabel('Price($)', fontsize=18)
    ax.set_xlabel('Dormitory Type', fontsize=18)
    
    fig = plt.gcf()
    st.pyplot(fig, use_container_width=True)
    
    return None




# @st.cache_data
def price_per_feature(data):
    # Define the options for the selectboxes
    options = ['bathrooms', 'bedrooms', 'house_age', 'basement', 'dormitory_type',  
               'floors', 'waterfront', 'view', 'condition', 'grade', 'standard', 'city']

    # Create the selectboxes
    feature_x = st.selectbox('Select the feature you would like to analyze on the x-axis', options, index=0)
    feature_y = st.selectbox('Select the feature you would like to analyze on the y-axis', options, index=1)
    
    # Check if feature_x and feature_y have the same value
    if feature_x == feature_y:
        st.error("Error: The selected features must be different.")
        return None
    # Create the pivot table
    pivot_table = data[['price', feature_x, feature_y]].pivot_table(index=feature_y, columns=feature_x, values='price', aggfunc='mean')

    # Create the heatmap
    fig, ax = plt.subplots(figsize=(30, 10))
    sns.heatmap(pivot_table, annot=True, cmap='cividis', fmt=".0f", ax=ax)
    
    # Set the title and labels
    ax.set_title(f"Average price by {feature_x} and {feature_y}", fontsize=25)
    ax.set_ylabel(f'{feature_y}', fontsize=18)
    ax.set_xlabel(f'{feature_x}', fontsize=18)
    
    # Display the plot
    st.pyplot(fig, use_container_width=True)
    
    return None

# ----------------------------- Start of the logical code structure -----------------------------

st.set_page_config(page_title='EDA',
                   page_icon='📈',
                   layout='wide'
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

st.sidebar.markdown('# Exploratory data analysis ')

st.sidebar.title('Commercial Options')
# ---------------Average price per year---------------
data["date"] = pd.to_datetime(data["date"]).dt.strftime('%Y-%m-%d')

min_year_built = int(data.yr_built.min())
max_year_built = int(data.yr_built.max())
st.sidebar.subheader('Select Max Year Built')
f_year_built = st.sidebar.slider('Year Built', 
                                 min_year_built,
                                 max_year_built,
                                 max_year_built)

# ---------------Average price per day---------------
st.sidebar.subheader('Select Max Date')
# filters
min_date = datetime.strptime(data.date.min(), '%Y-%m-%d')
max_date = datetime.strptime(data.date.max(), '%Y-%m-%d')

f_date = st.sidebar.slider('Date', 
                           min_date,
                           max_date,
                           max_date)

# ---------------Attributes Options---------------
st.sidebar.title('Attributes Options')
# filters
f_bedrooms = st.sidebar.selectbox('Max number of bedrooms',
                                  data.bedrooms.sort_values(ascending=False).unique())
f_bathooms = st.sidebar.selectbox('Max number of bathrooms',
                                  data.bathrooms.sort_values(ascending=False).unique())
f_floors = st.sidebar.selectbox('Max number of floor',
                                data.floors.sort_values(ascending=False).unique())
f_waterview = st.sidebar.checkbox('Only houses with water view')




st.sidebar.markdown('''___''')
st.sidebar.markdown('##### Powered by Comunidade DS')
st.sidebar.markdown('##### Data Analyst: Daniel Gomes')






# ==========================================================
#                Layout in Streamlit
# ==========================================================

with st.container():
    # Order Metric
    st.markdown('# Overall Metrics')
    col1, col2 = st.columns(2, gap='large')
    with col1:
        col1.markdown( '### Average metrics per city' )
        col1.write(avg_values(data).round(2))
    with col2:
        col2.markdown( '### Descriptive Analysis' )
        col2.write(data.describe().drop('count', axis=0).drop(['year_sold', 'yr_built', 'yr_renovated'], axis=1).T.round(2)) 
        
st.markdown( '## Region Overview' )
with st.container():
    col1, col2 = st.columns(2, gap='large')
    with col1:
        portfolio_density(data[['id', 'city', 'price', 'sqft_living', 'price_per_sqft', 'lat', 'long']], geofile)
    with col2:
        price_density_map(data[["price", "zipcode", 'lat', 'long']], geofile)

st.title('Commercial Attributes')
average_price_per_year(data[['yr_built', 'price', 'yr_renovated']], f_year_built)
average_price_per_day(data[['date', 'price']], f_date)

st.title( 'Houses Attributes')
attributes_distribution(data[['floors', 'bathrooms', 'bedrooms']], f_bedrooms, f_bathooms, f_floors, f_waterview)

st.title('Price disribution')
price_distribution(data[['lat', 'long', 'price', 'standard']])

st.title('To better understand the correlation between the variables, a correlation map was produced so that these behaviors can be analyzed')
correlation_map(data)

st.title('Information about property prices/sqft by dormitory type')
info_dormitory(data[['dormitory_type', 'price_per_sqft', 'price', 'standard']])

st.title('Average price per selected feature')
price_per_feature(data)